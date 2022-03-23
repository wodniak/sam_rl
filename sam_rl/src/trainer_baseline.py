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
import argparse
import numpy as np
import matplotlib.pyplot as plt
import itertools
import yaml

from stable_baselines3 import DDPG, TD3, SAC, PPO
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecMonitor,
    SubprocVecEnv,
    VecNormalize,
)
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import Figure

from env.env_eom_gym import EnvEOMGym


class TensorboardCallback(BaseCallback):
    """Plotting in tensorboard"""

    def __init__(self, action_dim, state_dim, ep_length, n_envs, verbose=1):
        super(TensorboardCallback, self).__init__(verbose)
        self.reward = 0
        self.ts = 0

        self.ep_length = ep_length
        self.actions = np.zeros([ep_length, action_dim])
        self.states = np.zeros([ep_length, state_dim])
        self.t = np.linspace(0, ep_length, ep_length).astype(int)
        self.episode = 0
        self.n_envs = n_envs

    def _on_rollout_end(self) -> None:
        self.logger.record("reward", self.reward)

        # if self.episode % self.n_envs == 0:
        #     self.plot_trim()
        #     self.plot_traj_3d()

        self.logger.dump(self.num_timesteps)
        self.reward = 0
        self.ts = 0
        self.episode += 1

    def _on_step(self) -> bool:
        reward = self.training_env.get_attr("reward")[0]
        # state = self.training_env.get_attr("full_state")[0]
        # action = self.training_env.get_attr("prev_action")[0]

        self.reward += reward
        # self.states[self.ts] = state
        # self.actions[self.ts] = action
        self.ts += 1

        return True

    def plot_trim(self) -> None:
        """dsds"""
        # fig, axs = plt.subplots(3)
        # axs[0].set_ylim([-1.1, 1.1])
        # axs[1].set_ylim([-1.1, 1.1])
        # axs[2].set_ylim([-1.1, 1.1])
        # axs[0].plot(self.t, self.actions[:, 0], label="rpm")
        # axs[1].plot(self.t, self.actions[:, 1:3], label=["de", "dr"])
        # axs[2].plot(self.t, self.actions[:, 3:5], label=["lcg", "vbs"])
        # for ax in axs:
        #     ax.legend()
        #     ax.grid()
        fig = plt.figure(1)
        plt.plot(self.t, self.actions, label=["lcg", "vbs"])
        plt.legend()
        plt.grid()
        self.logger.record(
            f"trim/{self.episode}_actions",
            Figure(fig, close=True),
            exclude=("stdout", "log", "json", "csv"),
        )
        plt.close()

        fig, axs = plt.subplots(2)
        axs[1].set_ylim([-1.6, 1.6])
        axs[0].plot(self.t, self.states[:, 0:3], label=["x", "y", "z"])  # z
        axs[1].plot(self.t, self.states[:, 3:6], label=["phi", "theta", "psi"])  # theta
        for ax in axs:
            ax.legend()
            ax.grid()
        self.logger.record(
            f"trim/{self.episode}_states",
            Figure(fig, close=True),
            exclude=("stdout", "log", "json", "csv"),
        )
        plt.close()

    def plot_traj_3d(self):
        """Plot 3D the trajectory"""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(
            self.states[0 : self.ep_length - 2, 0],
            self.states[0 : self.ep_length - 2, 1],
            self.states[0 : self.ep_length - 2, 2],
            "k-",
            label="sim",
        )
        ax.plot(
            self.states[:1, 0],
            self.states[:1, 1],
            self.states[:1, 2],
            "go",
            label="start",
        )
        ax.plot(
            self.states[-1, 0],
            self.states[-1, 1],
            self.states[-1, 2],
            "ro",
            label="end",
        )
        ax.plot(0.0, 0.0, 0.0, "ko", label="target")

        # format
        ax.set_xlabel("$x~[m]$")
        ax.set_ylabel("$y~[m]$")
        ax.set_zlabel("$z~[m]$")
        plt.legend()
        self.logger.record(
            f"trim/{self.episode}_traj_3d",
            Figure(fig, close=True),
            exclude=("stdout", "log", "json", "csv"),
        )
        plt.close()


class SaveVecNormalizeCheckpoint(BaseCallback):
    """Save VecNormalize into pkl during training"""

    def __init__(
        self, save_freq: int, save_path: str, name_prefix: str, verbose: int = 0
    ):
        super(SaveVecNormalizeCheckpoint, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self):
        if self.num_timesteps % self.save_freq == 0:
            print("Saving model...")
            if hasattr(self.model, "save"):
                model_path = "{}/{}_{}_steps".format(
                    self.save_path, self.name_prefix, self.num_timesteps
                )
                self.model.save(model_path)
            else:
                print(
                    f"[WARN] Model {self.name_prefix} not saved at timestep {self.num_timesteps}"
                )

            if hasattr(self.training_env, "save"):
                env_path = "{}/{}_{}_steps_env.pkl".format(
                    self.save_path, self.name_prefix, self.num_timesteps
                )
                self.training_env.save(env_path)
        return True


def make_env(i, ep_length, seed=0):
    """Utility function for multiprocessed env"""

    def _init():
        env = EnvEOMGym(ep_length, i)
        return env

    # set_global_seeds(seed)
    return _init


def train(model_type: str, params):
    """Start training"""

    ep_length = params["episode_length"]
    env_state_dict = params["env_state"]
    env_actions_dict = params["env_actions"]
    # env = SubprocVecEnv([make_env(i, ep_length) for i in range(params["num_cpu"])])
    # env = VecMonitor(venv=env)
    # env = VecNormalize(
    #     venv=env,
    #     training=True,
    #     norm_obs=True,
    #     norm_reward=False,
    #     clip_obs=40.0,
    #     gamma=0.99,
    # )

    env = EnvEOMGym(
        episode_length=params["episode_length"],
        dt=params["env_dt"],
        env_obs_states=params["env_state"],
        env_obs_state_reset=params["env_state_reset"],
        env_actions=params["env_actions"],
        env_reward_fn_type=params["env_reward_fn_type"],
        weights_Q=params["env_state_weights_Q"],
        weights_R=params["env_actions_weights_R"],
        weights_R_r=params["env_actions_weights_R_r"],
        num_envs=params["num_cpu"],
    )
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    # for evaluation
    # eval_env = SubprocVecEnv([make_env(i, ep_length) for i in range(1)])
    # eval_env = VecMonitor(venv=eval_env)
    # eval_env = VecNormalize(
    #     venv=eval_env,
    #     training=False,
    #     norm_obs=True,
    #     norm_reward=False,
    #     clip_obs=40.0,
    #     gamma=0.99,
    # )
    eval_env = EnvEOMGym(
        episode_length=params["episode_length"],
        dt=params["env_dt"],
        env_obs_states=params["env_state"],
        env_obs_state_reset=params["env_state_reset"],
        env_actions=params["env_actions"],
        env_reward_fn_type=params["env_reward_fn_type"],
        weights_Q=params["env_state_weights_Q"],
        weights_R=params["env_actions_weights_R"],
        weights_R_r=params["env_actions_weights_R_r"],
        num_envs=params["num_cpu"],
    )
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])

    # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions), sigma=params["sigma"] * np.ones(n_actions)
    )

    # Custom actor/critic architecture
    if model_type == "ddpg":
        model = DDPG(
            policy="MlpPolicy",
            env=env,
            action_noise=action_noise,
            verbose=params["verbose"],
            learning_rate=params["learning_rate"],
            buffer_size=params["buffer_size"],
            learning_starts=params["learning_starts"],
            batch_size=params["batch_size"],
            tau=params["tau"],
            gamma=params["gamma"],
            tensorboard_log=params["tensorboard_log"],
            device=params["device"],
            policy_kwargs=params["off_policy_kwargs"],
            # train_freq=params['train_freq'],
            # gradient_steps=params['gradient_steps']
        )
    elif model_type == "td3":
        model = TD3(
            policy="MlpPolicy",
            env=env,
            action_noise=action_noise,
            verbose=params["verbose"],
            learning_rate=params["learning_rate"],
            buffer_size=params["buffer_size"],
            learning_starts=params["learning_starts"],
            batch_size=params["batch_size"],
            tau=params["tau"],
            gamma=params["gamma"],
            tensorboard_log=params["tensorboard_log"],
            device=params["device"],
            policy_kwargs=params["off_policy_kwargs"],
            train_freq=params["train_freq"],
            gradient_steps=params["gradient_steps"],
        )
    elif model_type == "sac":
        model = SAC(
            policy="MlpPolicy",
            env=env,
            action_noise=action_noise,
            verbose=params["verbose"],
            learning_rate=params["learning_rate"],
            buffer_size=params["buffer_size"],
            learning_starts=params["learning_starts"],
            batch_size=params["batch_size"],
            tau=params["tau"],
            gamma=params["gamma"],
            tensorboard_log=params["tensorboard_log"],
            device=params["device"],
            policy_kwargs=params["off_policy_kwargs"],
            train_freq=params["train_freq"],
            gradient_steps=params["gradient_steps"],
        )
    elif model_type == "ppo":
        model = PPO(
            policy="MlpPolicy",
            env=env,
            policy_kwargs=params["on_policy_kwargs"],
            gamma=params["gamma"],
            learning_rate=params["learning_rate"],
            n_steps=params["episode_length"],
            batch_size=params["batch_size"],
            tensorboard_log=params["tensorboard_log"],
            verbose=params["verbose"],
        )

    print("Starting learning...")

    # read from env
    action_dim = env.get_attr("action_space", 0)[0].shape[-1]
    state_dim = env.get_attr("observation_space", 0)[0].shape[-1]

    # read from params
    num_cpu = params["num_cpu"]
    total_ts = ep_length * params["total_episodes"]
    save_freq = max(ep_length * params["save_freq"] // num_cpu, 1)
    eval_freq = max(ep_length * params["eval_freq"] // num_cpu, 1)
    n_eval_episodes = params["n_eval_episodes"]

    # Define all callbacks
    rewards_callback = TensorboardCallback(
        action_dim=action_dim,
        state_dim=state_dim,
        ep_length=ep_length,
        n_envs=num_cpu,
        verbose=params["verbose"],
    )

    checkpoint_callback = SaveVecNormalizeCheckpoint(
        save_freq=save_freq, save_path=model_path, name_prefix=model_type
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_path,
        log_path=tf_writer_path,
        eval_freq=eval_freq,
        deterministic=True,
    )

    callback_list = CallbackList([rewards_callback, checkpoint_callback, eval_callback])

    model.learn(
        total_timesteps=total_ts,
        callback=callback_list,
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        eval_log_path=tf_writer_path,
    )


def test(model_type: str, params):
    """Testing the model"""

    def plot_trim_with_setpoint(
        title: str, epoch, plot_dir: str, t, states, actions, t_setpoint
    ):
        """Plot states and actions"""
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        # actions
        fig, axs = plt.subplots(3)
        axs[0].set_ylim([-1.1, 1.1])
        axs[1].set_ylim([-1.1, 1.1])
        axs[2].set_ylim([-1.1, 1.1])
        axs[0].plot(t, actions[:, 0], label="rpm")
        axs[1].plot(t, actions[:, 1:3], label=["de", "dr"])
        axs[2].plot(t, actions[:, 3:5], label=["lcg", "vbs"])
        for ax in axs:
            ax.legend()
            ax.grid()

        fig.suptitle(title)
        plt.savefig(plot_dir + f"{epoch}_actions")
        plt.close()

        # states
        fig, axs = plt.subplots(2)
        axs[1].set_ylim([-1.6, 1.6])
        axs[0].plot(t, states[:, 0:3], label=["x", "y", "z"])  # z
        axs[0].plot(t, t_setpoint[:, 0], "--", label="x setpoint")  # x setpoint
        axs[0].plot(t, t_setpoint[:, 1], "--", label="y setpoint")  # y setpoint
        axs[0].plot(t, t_setpoint[:, 2], "--", label="z setpoint")  # z setpoint
        axs[1].plot(t, states[:, 3:6], label=["phi", "theta", "psi"])  # theta
        axs[1].plot(
            t, t_setpoint[:, 4], "k--", label="theta setpoint"
        )  # theta setpoint
        for ax in axs:
            ax.legend()
            ax.grid()

        fig.suptitle(title)
        plt.savefig(plot_dir + f"{epoch}_states")
        plt.close()

    def plot_traj_3d(title: str, epoch, plot_dir, t, states, setpoint):
        """3D trajectory plot"""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(states[:, 0], states[:, 1], states[:, 2], "k-", label="sim")
        ax.plot(states[:1, 0], states[:1, 1], states[:1, 2], "go", label="start")
        ax.plot(states[-1, 0], states[-1, 1], states[-1, 2], "ro", label="end")
        ax.plot(setpoint[0], setpoint[1], setpoint[2], "ko", label="setpoint")

        # format
        ax.set_xlabel("$x~[m]$")
        ax.set_ylabel("$y~[m]$")
        ax.set_zlabel("$z~[m]$")
        plt.legend()
        fig.suptitle(title)
        plt.savefig(plot_dir + f"{epoch}_traj_3d")

    def plot_traj_2d(title: str, epoch, plot_dir, t, states, setpoint):
        """2D XY trajectory plot"""
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(states[:, 0], states[:, 1], "k-", label="sim")
        ax.plot(states[:1, 0], states[:1, 1], "go", label="start")
        ax.plot(states[-1, 0], states[-1, 1], "ro", label="end")
        ax.plot(setpoint[0], setpoint[1], "ko", label="setpoint")

        # format
        ax.set_xlabel("$x~[m]$")
        ax.set_ylabel("$y~[m]$")
        plt.legend()
        fig.suptitle(title)
        plt.savefig(plot_dir + f"{epoch}_traj_2d")

    # model_name = "08.53.18-03.22.2022/td3_840000_steps.zip"  # blue 6d
    # model_name = "19.29.39-03.22.2022/td3_1290000_steps.zip"  # pink xy 6d
    # model_name = "19.26.44-03.22.2022/ppo_900000_steps.zip"  # light blue trim 6d
    model_name = "10.52.03-03.23.2022/td3_240000_steps.zip"  # orange 6d

    # model_name = "08.52.04-03.22.2022/td3_750000_steps.zip"  # gray 12d
    # model_name = "09.43.06-03.21.2022/td3_420000_steps.zip"  # blue 12d
    # model_name = "19.17.38-03.22.2022/td3_1230000_steps.zip"  # red 12d
    env_name = "13.36.38-03.15.2022/td3_1050000_steps_env.pkl"

    env_path = params["model_dir"] + env_name
    model_path = params["model_dir"] + model_name
    # model_path = "/home/gwozniak/catkin_ws/src/smarc_rl_controllers/sam_rl/baseline_logs_cache/2_test_xy_waypoint/td3_999750_steps.zip"
    # env_path = "/home/gwozniak/catkin_ws/src/smarc_rl_controllers/sam_rl/baseline_logs_cache/2_test_xy_waypoint/td3_999750_steps_env.pkl"

    env = EnvEOMGym(
        episode_length=params["test_episode_length"],
        dt=params["env_dt"],
        env_obs_states=params["env_state"],
        env_obs_state_reset=params["env_state_reset"],
        env_actions=params["env_actions"],
        env_reward_fn_type=params["env_reward_fn_type"],
        weights_Q=params["env_state_weights_Q"],
        weights_R=params["env_actions_weights_R"],
        weights_R_r=params["env_actions_weights_R_r"],
        num_envs=params["num_cpu"],
    )
    check_env(env)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    # env = VecNormalize(venv=env, training=False, norm_obs=True, norm_reward=False)
    env = VecNormalize.load(env_path, env)
    env.reset()

    assert os.path.exists(model_path), f"Model {model_path} does not exist."
    print(f"Loading model {model_path}...")
    if model_type == "ddpg":
        model = DDPG.load(path=model_path)
    elif model_type == "td3":
        model = TD3.load(path=model_path)
    elif model_type == "sac":
        model = SAC.load(path=model_path)
    elif model_type == "ppo":
        model = PPO.load(path=model_path)
    print(model.policy)

    max_episodes = 10
    ep_length = env.get_attr("ep_length", 0)[0]
    action_dim = env.get_attr("action_space", 0)[0].shape[-1]
    state_dim = env.get_attr("observation_space", 0)[0].shape[-1]

    # define setpoints
    if state_dim == 12:
        setpoints = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                # [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                # [0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                # [5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    # (x, z, theta, u, w, q)
    if state_dim == 6:
        setpoints = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.5, 0.0, 0.0, 0.0],
                [0.0, -3.0, 0.5, 0.0, 0.0, 0.0],
                [0.0, -3.0, -0.5, 0.0, 0.0, 0.0],
                [0.0, 10.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, -10.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

    num_setpoints = setpoints.shape[0]
    for episode in range(num_setpoints):
        # Reset env
        setpoint = setpoints[episode]
        obs = env.reset()
        start_state = obs
        end_state = obs

        # for plots
        ep_actions = np.zeros([ep_length, 5])
        ep_states = np.zeros([ep_length, 12])
        ep_t = np.linspace(0, ep_length, ep_length).astype(int)

        ep_reward = 0
        for ts in range(ep_length):
            obs -= setpoint  # will follow setpoint
            action, _states = model.predict(obs)  # deterministic for DDPG

            obs, rewards, dones, info = env.step(action)
            info = info[0]
            print(
                "[{}] {}\n{}\n{}\n{}\n".format(
                    ts,
                    setpoint,
                    info["state"],
                    info["actions"],
                    info["rewards"],
                )
            )
            end_state = obs
            ep_reward += rewards

            ep_actions[ts] = [*info["actions"].values()]  # save
            ep_states[ts] = list(itertools.chain(*info["state"].values()))  # save

        # after episode
        print(
            f"[{episode}] \tsetpoint = {setpoint}\n \
                            \tstart = {start_state}\n \
                            \tend = {end_state}\n \
                            \treward = {ep_reward}"
        )

        plot_test_dir = plot_dir + "test/"
        t_setpoint = np.tile(setpoint, (ep_length, 1))
        title = f"{model_type}:{model_name}"
        plot_trim_with_setpoint(
            title, episode, plot_test_dir, ep_t, ep_states, ep_actions, t_setpoint
        )
        plot_traj_3d(title, episode, plot_test_dir, ep_t, ep_states, setpoint)
        plot_traj_2d(title, episode, plot_test_dir, ep_t, ep_states, setpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM RL")
    parser.add_argument(
        "--model",
        dest="model",
        type=str,
        nargs="?",
        required=True,
        choices=["ddpg", "td3", "sac", "ppo"],
        help="Choose the model",
    )
    parser.add_argument(
        "--train",
        "-t",
        dest="train",
        action="store_true",
        help="Train new model if true",
    )
    parser.add_argument(
        "--env",
        "-e",
        dest="env",
        type=str,
        nargs="?",
        default="eom",
        choices=["eom", "stonefish"],
        help="Choose the environment",
    )
    parser.add_argument(
        "--config",
        "-c",
        dest="config",
        type=str,
        nargs="?",
        default="default",
        choices=["trim_6d", "trim_12d", "xy_6d", "xy_12d"],
        help="Choose the model",
    )
    args = parser.parse_args()

    assert args.model is not None, "Invalid argument for --model"
    assert args.env is not None, "Invalid argument for --env"

    # define directories
    base_dir = (
        os.path.expanduser("~")
        + "/catkin_ws/src/smarc_rl_controllers/sam_rl/baseline_logs/"
    )
    config_dir = (
        os.path.expanduser("~")
        + "/catkin_ws/src/smarc_rl_controllers/sam_rl/src/config/"
    )
    model_dir = base_dir + "model/"
    plot_dir = base_dir + "plots/"
    tensorboard_logs_dir = base_dir + "tensorboard_logs/"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir + "train/")
        os.makedirs(plot_dir + "test/")
    if not os.path.exists(tensorboard_logs_dir):
        os.makedirs(tensorboard_logs_dir)

    # define paths
    start_time = datetime.datetime.now().strftime("%H.%M.%S-%m.%d.%Y")
    tf_writer_path = tensorboard_logs_dir + start_time
    model_path = model_dir + start_time
    config_path = config_dir + args.config + ".yaml"

    with open(config_path) as params:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        parameter_dict = yaml.load(params, Loader=yaml.FullLoader)
        parameter_dict["tensorboard_log"] = tf_writer_path
        parameter_dict["model_dir"] = model_dir

    if args.env == "eom":
        if args.train:
            train(args.model, parameter_dict)
        else:
            test(args.model, parameter_dict)

    elif args.env == "stonefish":
        import trainer_stonefish

        model_name = "08.53.18-03.22.2022/td3_840000_steps.zip"  # blue 6d

        # model_name = "19.17.38-03.22.2022/td3_1230000_steps.zip"  # red 12d
        env_name = "15.53.43-03.16.2022/td3_999750_steps_env.pkl"

        env_path = model_dir + env_name
        model_path = model_dir + model_name

        trainer_stonefish.run_node(
            params=parameter_dict,
            model_path=model_path,
            model_type=args.model,
            train=args.train,
        )
