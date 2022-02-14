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
import argparse
import matplotlib.pyplot as plt
from torchinfo import summary

from stable_baselines3 import DDPG, TD3, SAC, PPO
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
        self.reward = self.training_env.get_attr("reward")[0] #only last reward
        self.logger.record("reward", self.reward)

        if self.episode % self.print_episode == 0:
            self.plot_trim()

        self.logger.dump(self.num_timesteps)
        self.reward = 0
        self.episode += 1

    def _on_step(self) -> bool:
        # self.reward += self.training_env.get_attr("reward")[0]
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

        for ax in axs:
            ax.legend()
            ax.grid()

        self.logger.record(f'trim/{self.episode}', Figure(fig, close=True), exclude=("stdout", "log", "json", "csv"))
        plt.close()

def make_env(i, seed=0):
    """Utility function for multiprocessed env"""
    def _init():
        env = EnvEOMGym(i)
        return env
    # set_global_seeds(seed)
    return _init

def train(model_type : str):
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
    # policy_kwargs = dict(net_arch=dict(pi=[32, 32], qf=[64, 128, 128]))
    policy_kwargs = dict(net_arch=dict(pi=[64, 64], qf=[64, 64])) # for off-policy only

    if model_type == 'ddpg':
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
    elif model_type == 'td3':
        model = TD3(policy="MlpPolicy",
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
    elif model_type == 'sac':
        model = SAC(policy="MlpPolicy",
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
    elif model_type == 'ppo':
        policy_kwargs = dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])]) # for on-policy only
        model = PPO(policy="MlpPolicy",
            env=env,
            tensorboard_log=tf_writer_path,
            policy_kwargs=policy_kwargs,
            verbose=1
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
        name_prefix=model_type)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_path,
        log_path=tf_writer_path,
        eval_freq=max(ep_length*100 // num_cpu, 1),
        deterministic=True
        )

    callback_list = CallbackList([
        rewards_callback,
        checkpoint_callback,
        eval_callback])

    total_ts = ep_length * 800

    model.learn(
        total_timesteps=total_ts,
        callback=callback_list,
        eval_env=eval_env,
        eval_freq=ep_length*50,
        n_eval_episodes=5,
        eval_log_path=tf_writer_path)
    # model.learn(total_timesteps=total_ts)
    model.save(model_path)

def test(model_type : str):
    def plot_trim_with_setpoint(title : str, epoch, plot_dir : str, t, states, actions, t_setpoint):
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        fig, axs = plt.subplots(3)
        axs[0].set_ylim([-1.1, 1.1])
        # axs[1].set_ylim([-5, 10])
        axs[2].set_ylim([-1.6, 1.6])
        axs[0].plot(t, actions, label = ['lcg', 'vbs']) # lcg, vbs
        axs[1].plot(t, states[:,1], label = 'z') # z
        axs[1].plot(t, t_setpoint[:,1], 'k--', label = 'z setpoint') # z setpoint
        axs[2].plot(t, states[:,2], label = 'theta') # theta
        axs[2].plot(t, t_setpoint[:,2], 'k--', label = 'theta setpoint') # theta setpoint

        for ax in axs:
            ax.legend()
            ax.grid()

        fig.suptitle(title)
        plt.savefig(plot_dir + f'{epoch}')

    env = EnvEOMGym()
    check_env(env)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    # model_name = '11.32.05-02.14.2022/ddpg_150000_steps.zip' # red ddpg
    # model_name = '10.41.47-02.14.2022/ddpg_150000_steps.zip' # gray ddpg

    # model_name = '13.04.18-02.14.2022/ddpg_240000_steps.zip' # blue ddpg
    model_name = '14.30.02-02.14.2022/ddpg_120000_steps.zip' # blue sac
    # model_name = '14.29.22-02.14.2022/ddpg_240000_steps.zip' # red td3
    # model_name = '14.59.37-02.14.2022/ddpg_240000_steps.zip' # pink ppo

    model_path = model_dir + model_name
    assert os.path.exists(model_path), f'Model {model_path} does not exist.'
    print(f'Loading model {model_path}...')
    if model_type == 'ddpg':
        model = DDPG.load(path=model_path)
    elif model_type == 'td3':
        model = TD3.load(path=model_path)
    elif model_type == 'sac':
        model = SAC.load(path=model_path)
    elif model_type == 'ppo':
        model = PPO.load(path=model_path)
    print(model.policy)

    max_episodes = 10
    ep_length = env.get_attr('ep_length', 0)[0]
    action_dim = env.get_attr('action_space', 0)[0].shape[-1]
    state_dim = env.get_attr('observation_space', 0)[0].shape[-1]

    #define setpoints
    setpoints = np.array([
        [0., 0.0, 0.0, 0., 0., 0.],
        [0., 5.0, 0.0, 0., 0., 0.],
        [0., 5.0, 0.5, 0., 0., 0.],
        [0., -3.0, 0.5, 0., 0., 0.],
        [0., -3.0, -0.5, 0., 0., 0.],
        [0., 10.0, 0.0, 0., 0., 0.],
        [0., -10.0, 0.0, 0., 0., 0.]
        ])

    for episode in range(setpoints.shape[-1]):
        # Reset env
        setpoint = setpoints[episode]
        obs = env.reset()
        start_state = obs
        end_state = obs

        # for plots
        actions = np.zeros([ep_length, action_dim])
        states = np.zeros([ep_length, state_dim])
        t = np.linspace(0, ep_length, ep_length).astype(int)

        ep_reward = 0
        for ts in range(ep_length):
            states[ts] = obs    # save

            obs -= setpoint     # will follow setpoint
            action, _states = model.predict(obs) # deterministic for DDPG

            actions[ts] = action    # save

            obs, rewards, dones, info = env.step(action)
            end_state = obs
            ep_reward += rewards

        # after episode
        print(f'[{episode}]  setpoint = {setpoint} start = {start_state} end = {end_state}  reward = {ep_reward}')

        plot_test_dir = plot_dir + 'test/'
        t_setpoint = np.tile(setpoint,(ep_length,1))
        title = f'{model_type}:{model_name}'
        plot_trim_with_setpoint(title, episode, plot_test_dir, t, states, actions, t_setpoint)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SAM RL')
    parser.add_argument('--model',
        dest='model',
        type=str,
        nargs='?',
        required=True,
        choices=['ddpg', 'td3', 'sac', 'ppo'],
        help='Choose the model')

    args = parser.parse_args()

    assert args.model is not None, 'Invalid argument for --model'

    # train(args.model)
    test(args.model)