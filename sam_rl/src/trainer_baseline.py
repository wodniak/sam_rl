#!/usr/bin/env python3

# Copyright 2022 Grzegorz Wozniak (gwozniak@kth.se)
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided
# that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
#    following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
#    following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
#  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
#  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
#  OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import os
import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, SubprocVecEnv, vec_check_nan
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import Figure

from env.env_eom_gym import EnvEOMGym


# define global paths
base_dir = os.path.expanduser('~') + '/catkin_ws/src/smarc_rl_controllers/sam_rl/baseline_logs/'
model_dir = base_dir + 'model/'
plot_dir = base_dir + 'plots/'
tensorboard_logs_dir = base_dir + 'tensorboard_logs/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir + 'train/')
    os.makedirs(plot_dir + 'test/')
if not os.path.exists(tensorboard_logs_dir):
    os.makedirs(tensorboard_logs_dir)


class TensorboardCallback(BaseCallback):
    def __init__(self, action_dim, state_dim, ep_length, n_envs, verbose=1):
        super(TensorboardCallback, self).__init__(verbose)
        self.reward = 0

        # hardcoded size depending on the env
        self.actions = np.zeros([ep_length, action_dim])
        self.states = np.zeros([ep_length, state_dim])
        self.t = np.linspace(0, ep_length, ep_length).astype(int)
        self.episode = 0
        self.print_episode = 20 * n_envs

    def _on_rollout_end(self) -> None:
        self.logger.record("reward", self.reward)

        if self.episode % self.print_episode == 0:
            self.plot_trim()

        self.logger.dump(self.num_timesteps)
        self.reward = 0
        self.episode += 1

    def _on_step(self) -> bool:
        self.reward += self.training_env.get_attr("reward")[0]
        ts = self.training_env.get_attr("current_step")[0]
        self.states[ts] = self.training_env.get_attr("state")[0]
        self.actions[ts] = self.training_env.get_attr("prev_action")[0]
        return True

    def plot_trim(self) -> None:
        # (x, z, theta, u, w, q)
        fig, axs = plt.subplots(3)
        axs[0].set_ylim([-1.1, 1.1])
        # axs[1].set_ylim([-5, 10])
        axs[2].set_ylim([-1.6, 1.6])
        axs[0].plot(self.t, self.actions, label = ['lcg', 'vbs']) # lcg, vbs
        axs[1].plot(self.t, self.states[:,1], label = 'z') # z
        axs[2].plot(self.t, self.states[:,2], label = 'theta') # theta
        axs[0].legend()
        axs[1].legend()
        axs[2].legend()
        self.logger.record(f'trim/{self.episode}', Figure(fig, close=True), exclude=("stdout", "log", "json", "csv"))
        plt.close()

def make_env(i, seed=0):
    """Utility function for multiprocessed env"""
    def _init():
        env = EnvEOMGym(i)
        return env
    # set_global_seeds(seed)
    return _init

def train():
    # Set up tensorboard logging
    start_time = datetime.datetime.now().strftime("%H.%M.%S-%m.%d.%Y")
    tf_writer_path = tensorboard_logs_dir + start_time

    model_name = start_time
    model_path = model_dir + model_name

    env = EnvEOMGym()
    check_env(env)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    # for evaluation
    eval_env = EnvEOMGym()
    check_env(eval_env)
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])

    num_cpu = 1
    # env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    # env = VecMonitor(venv=env)

    # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # Custom actor/critic architecture
    policy_kwargs = dict(net_arch=dict(pi=[32, 32], qf=[64, 64, 128]))
    model = DDPG(policy="MlpPolicy",
                env=env,
                action_noise=action_noise,
                verbose=1,
                learning_rate=0.001,
                buffer_size=int(1e6),
                learning_starts=128,
                batch_size=128,
                tau=0.005,
                gamma=0.99,
                tensorboard_log=tf_writer_path,
                device='auto',
                policy_kwargs=policy_kwargs
                # train_freq=5,
                # gradient_steps=-1
                )

    print('Starting learning...')
    # action_dim=env.envs[0].action_space.shape[-1]
    # state_dim= env.envs[0].observation_space.shape[-1]
    # ep_length=env.envs[0].ep_length

    action_dim = env.get_attr('action_space', 0)[0].shape[-1]
    state_dim = env.get_attr('observation_space', 0)[0].shape[-1]
    ep_length = env.get_attr('ep_length', 0)[0]

    # Define all callbacks
    rewards_callback = TensorboardCallback(
        action_dim=action_dim,
        state_dim=state_dim,
        ep_length=ep_length,
        n_envs=num_cpu,
        verbose=1)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(ep_length*100 // num_cpu, 1),
        save_path=model_path,
        name_prefix='ddpg')

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_path,
        log_path=tf_writer_path,
        eval_freq=max(ep_length*100 // num_cpu, 1),
        deterministic=True
        )

    callback_list = CallbackList([rewards_callback, checkpoint_callback])

    total_ts = ep_length * 500

    model.learn(
        total_timesteps=total_ts,
        callback=callback_list,
        eval_env=eval_env,
        eval_freq=ep_length*50,
        n_eval_episodes=5,
        eval_log_path=tf_writer_path)
    # model.learn(total_timesteps=total_ts)
    model.save(model_path)

def test():
    def plot_trim(epoch, plot_dir, t, states, actions):
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        fig, axs = plt.subplots(3)
        axs[0].set_ylim([-1.1, 1.1])
        # axs[1].set_ylim([-5, 10])
        axs[2].set_ylim([-1.6, 1.6])
        axs[0].plot(t, actions, label = ['lcg', 'vbs']) # lcg, vbs
        axs[1].plot(t, states[:,1], label = 'z') # z
        axs[2].plot(t, states[:,2], label = 'theta') # theta
        axs[0].legend()
        axs[1].legend()
        axs[2].legend()
        plt.savefig(plot_dir + f'{epoch}')

    env = EnvEOMGym()
    check_env(env)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    model_name = '11.47.18-02.10.2022'
    model_path = model_dir + model_name
    assert os.path.exists(model_path), f'Model {model_path} does not exist.'
    print(f'Loading model {model_path}...')
    model = DDPG.load(model_path)

    max_episodes = 10
    ep_length = env.get_attr('ep_length', 0)[0]
    action_dim = env.get_attr('action_space', 0)[0].shape[-1]
    state_dim = env.get_attr('observation_space', 0)[0].shape[-1]

    obs = env.reset()
    for episode in range(max_episodes):
        actions = np.zeros([ep_length, action_dim])
        states = np.zeros([ep_length, state_dim])
        t = np.linspace(0, ep_length, ep_length).astype(int)
        ep_reward = 0
        for ts in range(ep_length):
            action, _states = model.predict(obs) # deterministic for DDPG
            obs, rewards, dones, info = env.step(action)

            # save
            ep_reward += rewards
            states[ts] = obs
            actions[ts] = action

        # after episode
        print(f'[{episode}]  reward = {ep_reward}')
        plot_test_dir = plot_dir + 'test/'
        plot_trim(episode, plot_test_dir, t, states, actions)

if __name__ == "__main__":
    train()
    # test()