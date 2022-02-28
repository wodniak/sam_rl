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
import numpy as np

import rospy
from env.env_stonefish import SAMEnv

from stable_baselines3 import TD3


def test(
    model_type : str,
    model_path : str,
    env_path : str,
    env : SAMEnv,
    max_timesteps : int,
    setpoints : np.array
    ):

    assert os.path.exists(model_path), f'Model {model_path} does not exist.'
    assert os.path.exists(env_path), f'Env {env_path} does not exist.'
    assert setpoints.shape[1] == 12, f'Setpoint state dimensions are incorrect (should be 12)'

    print(f'Loading model {model_path}...')
    print(f'Loading env statistics {env_path}...')
    if model_type == 'ddpg':
        model = DDPG.load(path=model_paths)
    elif model_type == 'td3':
        model = TD3.load(path=model_path)
    elif model_type == 'sac':
        model = SAC.load(path=model_path)
    elif model_type == 'ppo':
        model = PPO.load(path=model_path)
    print(model.policy)


    episode = 0
    while True:
        print('New Episode')

        # Reset env
        setpoint = env.current_setpoint
        obs = env.reset()
        start_state = obs
        end_state = obs

        # for plots
        actions = np.zeros([max_timesteps, env.action_dim])
        states = np.zeros([max_timesteps, env.state_dim])
        t = np.linspace(0, max_timesteps, max_timesteps).astype(int)

        ep_reward = 0
        for ts in range(max_timesteps):
            states[ts] = obs    # save

            obs -= setpoint     # will follow setpoint
            action, _states = model.predict(obs)

            actions[ts] = action    # save

            obs, rewards, dones, info = env.step(action)
            end_state = obs
            ep_reward += rewards

        # after episode
        # print(f'[{episode}]\n \tsetpoint = {setpoint}\n \
        #                       \tstart = {start_state}\n \
        #                       \tend = {end_state}\n \
        #                       \treward = {ep_reward}')


if __name__ == "__main__":
    rospy.init_node("rl_trainer")
    model_type = rospy.get_param("~model_type", default="td3")

    # define paths
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

    model_name = '10.27.28-02.17.2022/td3_999750_steps.zip'
    env_name = '10.27.28-02.17.2022/td3_999750_steps_env.pkl'

    env_path = model_dir + env_name
    model_path = model_dir + model_name

    env = SAMEnv()

    #define setpoints
    setpoints = np.array([
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        ])
        # [0., 5., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        # [5., 5., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        # [-5., 5., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        # [5., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        # ])

    test(model_type=model_type,
         model_path=model_path,
         env_path=env_path,
         env=env,
         max_timesteps=500,
         setpoints=setpoints)