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

# Script that trains Agent DDPG on continuous pendulum environment
# Training results OK - Episode * 99 * Avg Reward is ==> -176.63040818927897
# @see https://keras.io/examples/rl/ddpg_pendulum/

import gym
import numpy as np
import matplotlib.pyplot as plt
from agent_ddpg import *

if __name__ == '__main__':

    problem = "Pendulum-v1"
    env = gym.make(problem)

    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    upper_bound = env.action_space.high[0]
    lower_bound = env.action_space.low[0]

    print("Size of State Space ->  {}".format(num_states))
    print("Size of Action Space ->  {}".format(num_actions))
    print("Max Value of Action ->  {}".format(upper_bound))
    print("Min Value of Action ->  {}".format(lower_bound))

    upper_bound = np.array([upper_bound])

    # hyperparams
    std_dev = 0.2
    critic_lr = 0.002    # Learning rate for actor-critic models
    actor_lr = 0.001
    total_episodes = 100
    gamma = 0.99     # Discount factor for future rewards
    tau = 0.005     # Used to update target networks
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    buffer_size = 50000

    ou_noise = OUActionNoise(
        mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

    agent = DDPGAgent(state_dim=num_states,
                      action_dim=num_actions,
                      max_action=upper_bound,
                      device=device,
                      discount=gamma,
                      tau=tau,
                      actor_lr=actor_lr,
                      critic_lr=critic_lr)

    buffer = ReplayBuffer(num_states, num_actions, buffer_size)

    # train
    ep_reward_list = []        # To store reward history of each episode
    avg_reward_list = []    # To store average reward history of last few episodes

    # Takes about 1 min to train
    for ep in range(total_episodes):

        prev_state = env.reset()
        episodic_reward = 0

        while True:
            # if ep == total_episodes - 1:
            env.render()

            action = agent.select_action(prev_state)
            action += ou_noise()

            # Recieve state and reward from environment.
            state, reward, done, info = env.step(action)

            buffer.add(prev_state, action, state, reward, done)
            episodic_reward += reward

            agent.train(buffer, batch_size)

            # End this episode when `done` is True
            if done:
                break

            prev_state = state

        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)

    # Plotting graph
    # Episodes versus Avg. Rewards
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()
