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

from torch.utils.tensorboard import SummaryWriter
from agents.agent_ddpg import DDPGAgent, ReplayBuffer, OUActionNoise
from env.env_stonefish import SAMEnv
from env.env_eom import EnvEOM_Task_Trim

# ROS
import rospy
from nav_msgs.msg import Path, Odometry

# SMARC
from smarc_msgs.msg import ThrusterRPM
from sam_msgs.msg import ThrusterAngles, PercentStamped
from std_msgs.msg import Float64, Header, Bool


class Trainer(object):
    def __init__(self, node_name):
        # Parameters
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        # Init environment
        # self.env = SAMEnv()
        self.env = EnvEOM_Task_Trim()

        # Learning parameters
        self.actor_lr = 0.001
        self.critic_lr = 0.002
        self.tau = 0.005
        self.discount = 0.99
        self.replay_buffer_max_size = int(5e5)
        self.batch_size = 128
        self.std_dev = 5 * np.ones(self.env.action_dim)
        self.expl_noise = OUActionNoise(mean=np.zeros(
            self.env.action_dim), std_deviation=self.std_dev)

        self.train_epoch = 200
        # self.max_timesteps = 6000  # per epoch
        self.max_timesteps = 400  # per epoch

        # Init RL agent
        self.agent = DDPGAgent(state_dim=self.env.state_dim,
                               action_dim=self.env.action_dim,
                               max_action=self.env.max_action,
                               device=self.device,
                               discount=self.discount,
                               tau=self.tau,
                               actor_lr=self.actor_lr,
                               critic_lr=self.critic_lr)

        # Init Memory
        self.replay_buffer = ReplayBuffer(
            state_dimension=self.env.state_dim,
            action_dimension=self.env.action_dim,
            max_size=self.replay_buffer_max_size)

        # Load model weights and metadata if exist
        self.agent_dir = os.path.expanduser('~') + '/.ros'
        model_name = f'09.26.33-02.03.2022'
        model = f'{self.agent_dir}/{model_name}'
        if os.path.exists(model + '_critic'):
            rospy.loginfo(f'Loading model {model}')
            # self.start_time, self.last_episode =
            self.agent.load_checkpoint(model)
            self.start_time = model_name
            self.last_episode = 83
        else:
            rospy.loginfo('No model found. Training from the beginning...')
            self.start_time = datetime.datetime.now().strftime("%H.%M.%S-%m.%d.%Y")
            self.last_episode = -1

        # Set up tensorboard logging
        self.tf_writer = SummaryWriter(os.path.join('logs', self.start_time))

    def train(self):
        """
        Training loop
        """
        # book-keeping
        evaluations = []
        epoch_rewards = []

        for epoch in range(self.last_episode + 1, self.last_episode + self.train_epoch + 1):
            state = self.env.reset()  # it should reset to initial state here
            # state = self.env.get_observation()
            epoch_reward = 0
            done = False

            ts = 0
            while ts < self.max_timesteps:
                # Calculate action
                action = self.agent.select_action(state)
                action += self.expl_noise()  # exploration
                # rospy.loginfo_throttle(0.1, action)
                np.clip(action, 0, self.env.max_action[0], out=action)

                # Make action
                next_state, reward, done = self.env.step(action)

                # Record it
                self.replay_buffer.add(
                    state, action, next_state, reward, done)

                # Train agent
                self.agent.train(self.replay_buffer, self.batch_size)

                state = next_state
                epoch_reward += reward

                ts += 1
                if done:
                    break

            # After each epoch
            epoch_rewards.append(epoch_reward)
            rospy.loginfo('Epoch: {} Steps: {} Reward: {}'.format(
                epoch, ts, epoch_reward))

            self.tf_writer.add_scalar('reward', epoch_reward, epoch)
            if epoch % 10 == 0:
                self.agent.save_checkpoint(self.start_time)

    def test(self):
        evaluations = []
        epoch_rewards = []
        import time

        test_epochs = 100
        for epoch in range(0, test_epochs):
            state = self.env.reset()  # it should reset to initial state here
            # state = self.env.get_observation()
            epoch_reward = 0
            done = False

            ts = 0
            while ts < self.max_timesteps:
                # Calculate action
                action = self.agent.select_action(state)
                # action += self.expl_noise()  # exploration
                np.clip(action, 0, self.env.max_action[0], out=action)

                rospy.loginfo(f'state = {state.round(2)}, action = {action.round(2)}')
                time.sleep(0.5)

                # Make action
                next_state, reward, done = self.env.step(action)
                state = next_state
                epoch_reward += reward

                ts += 1
                if done:
                    break

            # After each epoch
            epoch_rewards.append(epoch_reward)
            print()
            print()
            rospy.loginfo('--------------Epoch: {} Steps: {} Reward: {}'.format(
                epoch, ts, epoch_reward))


if __name__ == "__main__":
    rospy.init_node("rl_trainer")
    trainer = Trainer(rospy.get_name())
    # trainer.train()
    trainer.test()