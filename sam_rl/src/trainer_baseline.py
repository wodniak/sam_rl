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
import itertools
import yaml
import numpy as np

from stable_baselines3 import DDPG, TD3, SAC, PPO
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecNormalize,
    VecEnv,
)
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor

from env.env_eom_gym import EnvEOMGym
import utils.callbacks
import utils.test_plot


def init_env(params) -> EnvEOMGym:
    """
    :return Instantiated environment
    """
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
    return env


def make_subproc_env(params, seed=0):
    """Utility function for multiprocessed env"""

    def _init():
        env = init_env(params)
        return env

    # set_global_seeds(seed)
    return _init


def setup_vec_env(params, is_eval_env=False, norm_env_load_path=None) -> VecEnv:
    """
    Prepare EOM environment for training
    :return Wrapped EOM environment
    """
    num_cpu = params["num_cpu"]
    if num_cpu == 1:
        # No parallel envs, use Dummy vec Env.
        env = init_env(params)
        env = Monitor(env)
        vec_env = DummyVecEnv([lambda: env])
    else:
        # Init parallel envs
        # Create just one env if its for evaluation
        if is_eval_env:
            num_cpu = 1
            params["num_cpu"] = 1
        vec_env = SubprocVecEnv([make_subproc_env(params) for i in range(num_cpu)])
        vec_env = VecMonitor(venv=vec_env)

    normalize = params["env_use_vecnormalize"]
    if normalize:
        if norm_env_load_path is None:
            vecnormalize_params = params["vecnormalize"]
            vec_env = VecNormalize(
                venv=vec_env,
                training=vecnormalize_params["training"],
                norm_obs=vecnormalize_params["norm_obs"],
                norm_reward=vecnormalize_params["norm_reward"],
                clip_obs=vecnormalize_params["clip_obs"],
                gamma=vecnormalize_params["gamma"],
            )
        else:
            print(f"Loading Vec Env: {norm_env_load_path}")
            vec_env = VecNormalize.load(load_path=norm_env_load_path, venv=vec_env)
    return vec_env


def train(model_type: str, params):
    """Start training"""

    ep_length = params["episode_length"]

    env = setup_vec_env(params)
    eval_env = setup_vec_env(params, is_eval_env=True)

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
            train_freq=params["train_freq"],
            gradient_steps=params["gradient_steps"],
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
    rewards_callback = utils.callbacks.TensorboardCallback(
        action_dim=action_dim,
        state_dim=state_dim,
        ep_length=ep_length,
        n_envs=num_cpu,
        verbose=params["verbose"],
    )

    checkpoint_callback = utils.callbacks.SaveVecNormalizeCheckpoint(
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


def test(model_type: str, model_name, params):
    """Testing the model"""
    use_env = params["env_use_vecnormalize"]
    env_name = model_name[:-4] + "_env.pkl"

    env_path = params["model_dir"] + env_name
    model_path = params["model_dir"] + model_name

    use_env = False
    model_path = "/home/gwozniak/catkin_ws/src/smarc_rl_controllers/sam_rl/baseline_logs_cache/08.52.04-03.22.2022/td3_750000_steps.zip"
    # model_path = "/home/gwozniak/catkin_ws/src/smarc_rl_controllers/sam_rl/baseline_logs_cache/08.53.18-03.22.2022/td3_840000_steps.zip"
    # model_path = "/home/gwozniak/catkin_ws/src/smarc_rl_controllers/sam_rl/baseline_logs_cache/07.29.57-04.12.2022/td3_720000_steps.zip"
    # model_path = "/home/gwozniak/catkin_ws/src/smarc_rl_controllers/sam_rl/baseline_logs_cache/1_test_trim/td3_trim_6d_test.zip"
    # env_path = "/home/gwozniak/catkin_ws/src/smarc_rl_controllers/sam_rl/baseline_logs_cache/2_test_xy_waypoint/td3_999750_steps_env.pkl"

    # change normal parameters for test parameters
    params["episode_length"] = params["test_episode_length"]
    params["env_dt"] = params["test_env_dt"]

    if use_env:
        env = setup_vec_env(params, is_eval_env=True, norm_env_load_path=env_path)
    else:
        env = setup_vec_env(params, is_eval_env=True)
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

    ep_length = env.get_attr("ep_length", 0)[0]
    action_dim = env.get_attr("action_space", 0)[0].shape[-1]
    state_dim = env.get_attr("observation_space", 0)[0].shape[-1]

    # load setpoints
    setpoints = params["test_setpoints"]
    setpoints = np.array(setpoints)

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
                "{}-[{}] {}\n{}\n{}\n{}\n".format(
                    episode,
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

        # utils.test_plot.save_in_csv(
        #     title, episode, plot_test_dir, ep_t, ep_states, ep_actions, t_setpoint
        # )
        utils.test_plot.plot_trim_with_setpoint(
            title, episode, plot_test_dir, ep_t, ep_states, ep_actions, t_setpoint
        )
        utils.test_plot.plot_traj_3d(
            title, episode, plot_test_dir, ep_t, ep_states, setpoint
        )
        utils.test_plot.plot_traj_2d(
            title, episode, plot_test_dir, ep_t, ep_states, setpoint
        )


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
        choices=[
            "trim_6d",
            "trim_12d",
            "xy_6d",
            "xy_12d",
            "pendulum",
            "tight_turn_6d",
            "tight_turn_12d",
        ],
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

    start_time = datetime.datetime.now().strftime("%H.%M.%S-%m.%d.%Y")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir + "train/")
        os.makedirs(plot_dir + "test/")
    if not os.path.exists(tensorboard_logs_dir):
        os.makedirs(tensorboard_logs_dir)

    # PICK MODEL HERE
    # model_name = "08.31.38-05.09.2022/td3_680000_steps.zip"  # trim
    # model_name = "08.32.14-05.09.2022/td3_600000_steps.zip"  # trim
    # model_name = "08.39.39-05.09.2022/td3_680000_steps.zip"  # trim
    # model_name = "08.42.20-05.09.2022/ppo_840000_steps.zip"  # trim ppo
    # model_name = "08.42.58-05.09.2022/ppo_300000_steps.zip"  # trim ppo
    # model_name = "13.20.25-05.11.2022/td3_1640000_steps.zip"  # trim

    # model_name = "08.53.18-03.22.2022/td3_840000_steps.zip"
    model_name = "08.52.04-03.22.2022/td3_750000_steps.zip"

    # model_name = "13.03.00-05.03.2022/td3_1880000_steps.zip"  # xy GOOD
    # model_name = "13.02.29-05.03.2022/td3_1960000_steps.zip"  # xy
    # model_name = "06.23.29-05.05.2022/td3_1080000_steps.zip"  # xy

    # model_name = "15.22.52-04.26.2022/ddpg_800000_steps.zip"  # pendulum
    # model_name = "13.19.56-05.11.2022/td3_2000000_steps.zip"  # pendulum

    # define paths
    tf_writer_path = tensorboard_logs_dir + start_time
    model_path = model_dir + start_time

    # config_path = config_dir + args.config + ".yaml"
    if args.train:
        config_path = config_dir + args.config + ".yaml"
    else:
        date = model_name.split("/")[0]
        config = args.config + ".yaml"
        config_path = model_dir + date + "/" + config

    with open(config_path) as params:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        print(f"Loading config file : {config_path}")
        parameter_dict = yaml.load(params, Loader=yaml.FullLoader)
        parameter_dict["tensorboard_log"] = tf_writer_path
        parameter_dict["model_dir"] = model_dir

    if args.env == "eom":
        if args.train:
            if not os.path.exists(model_dir + start_time):
                os.makedirs(model_dir + start_time)
            # save the params that started the training in the model folder
            params_save_path = model_dir + start_time + "/" + args.config + ".yaml"
            with open(params_save_path, "w") as file:
                documents = yaml.dump(parameter_dict, file)
            train(args.model, parameter_dict)
        else:
            test(args.model, model_name, parameter_dict)

    elif args.env == "stonefish":
        import trainer_stonefish

        # model_name = "10.10.12-03.24.2022/td3_1200000_steps.zip"  #
        env_name = "10.10.12-03.24.2022/td3_1200000_steps_env.pkl"

        env_path = model_dir + env_name
        model_path = model_dir + model_name
        trainer_stonefish.run_node(
            params=parameter_dict,
            model_path=model_path,
            model_type=args.model,
            train=args.train,
        )
