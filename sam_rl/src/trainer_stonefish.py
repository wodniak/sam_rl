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
import argparse
import itertools
import rospy
import numpy as np
from env.env_stonefish import SAMEnv
from stable_baselines3 import TD3, DDPG, SAC, PPO


def test(model_type: str, model_path: str, env: SAMEnv, max_timesteps: int):
    """Run model in stonefish"""

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

    episode = 0
    while True:
        print("New Episode")

        # Reset env
        setpoint = env.get_current_setpoint()
        obs = env.reset()

        # for plots
        ep_actions = np.zeros([max_timesteps, 5])
        ep_states = np.zeros([max_timesteps, 12])
        ep_t = np.linspace(0, max_timesteps, max_timesteps).astype(int)

        ep_reward = 0
        for ts in range(max_timesteps):
            obs -= setpoint  # will follow setpoint
            action, _states = model.predict(obs)

            obs, rewards, dones, info = env.step(action)
            end_state = obs
            ep_reward += rewards

            ep_actions[ts] = [*info["actions"].values()]  # save
            ep_states[ts] = list(itertools.chain(*info["state"].values()))  # save

            print(
                "[{}] {}\n{}\n{}\n{}\n{}\n".format(
                    ts, setpoint, info["state"], info["actions"], info["rewards"], obs
                )
            )

        # after episode
        # print(f'[{episode}]\n \tsetpoint = {setpoint}\n \
        #                       \tstart = {start_state}\n \
        #                       \tend = {end_state}\n \
        #                       \treward = {ep_reward}')


def run_node(params, model_path, model_type, train=False):
    """Start the ROS node for SAM"""
    assert os.path.exists(model_path), f"Model {model_path} does not exist."
    rospy.init_node("rl_trainer")
    env = SAMEnv(env_obs_states=params["env_state"], env_actions=params["env_actions"])

    if train:
        raise NotImplementedError(
            "Training in stonefish environment is not implemented."
        )
    else:
        test(
            model_type=model_type,
            model_path=model_path,
            env=env,
            max_timesteps=500,
        )
