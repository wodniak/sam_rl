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
import time
import matplotlib.pyplot as plt

# RL
from torch.utils.tensorboard import SummaryWriter
from agents.agent_ddpg import DDPGAgent, ReplayBuffer, OUActionNoise
from env.env_stonefish import SAMEnv
from env.env_eom import EnvEOM_Task_Trim, EnvEOM_Task_XY

# ROS
import rospy
from nav_msgs.msg import Path, Odometry

# SMARC
from smarc_msgs.msg import ThrusterRPM
from sam_msgs.msg import ThrusterAngles, PercentStamped
from std_msgs.msg import Float64, Header, Bool


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

def plot_xy(epoch, plot_dir, t, states, actions):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    fig, axs = plt.subplots(3)
    axs[0].set_ylim([-1.1, 1.1])
    # axs[1].set_ylim([-5, 10])
    # axs[2].set_ylim([-1.6, 1.6])
    axs[0].plot(t, actions, label = ['rpm', 'dr']) # rpm, dr
    axs[1].plot(t, states[:,0:2], label = ['x', 'y']) # x, y
    axs[2].plot(t, states[:,2], label = 'psi') # psi
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    plt.savefig(plot_dir + f'{epoch}')

def plot_traj_2d(epoch, plot_dir, states):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(states[:,0], states[:,1], 'k-', label='sim')
    ax.plot(states[:1,0], states[:1,1], 'ko', label=None)

    # format
    ax.set_xlabel('$x~[m]$')
    ax.set_ylabel('$y~[m]$')
    plt.legend()
    plt.savefig(plot_dir + f'traj_2d_{epoch}')

def plot_traj_3d(epoch, plot_dir, states):
    fig = plt.figure()
    ax = fig.add_subplot(1, projection='3d')

    ax.plot(traj[:,0], traj[:,1], traj[:,2], 'k-', label='sim')
    ax.plot(traj[:1,0], traj[:1,1], traj[:1,2], 'ko', label=None)

    # format
    ax.set_xlabel('$x~[m]$')
    ax.set_ylabel('$y~[m]$')
    ax.set_zlabel('$z~[m]$')
    plt.legend()
    plt.savefig(plot_dir + f'traj_3d_{epoch}')


class Trainer(object):
    def __init__(self, node_name):
        # Parameters
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        # Init environment
        # self.env = SAMEnv()
        self.env = EnvEOM_Task_Trim()
        self.env = EnvEOM_Task_XY()

        # Learning parameters
        self.actor_lr = 0.001
        self.critic_lr = 0.002
        self.tau = 0.005
        self.discount = 0.99
        self.replay_buffer_max_size = int(5e5)
        self.batch_size = 128
        # self.std_dev = 5 * np.ones(self.env.action_dim)   # for stonefish
        self.std_dev = 0.1 * np.ones(self.env.action_dim)   # for eom sim

        self.expl_noise = OUActionNoise(mean=np.zeros(
            self.env.action_dim), std_deviation=self.std_dev)

        self.train_epoch = 1000
        # self.max_timesteps = 6000  # per epoch
        self.max_timesteps = 300  # per epoch

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
        self.path_dir = os.path.expanduser('~') + '/catkin_ws/src/smarc_rl_controllers/sam_rl/logs'
        if not os.path.exists(self.path_dir):
            os.makedirs(self.path_dir)

        agent_dir = self.path_dir + '/model/'
        # model_name = f'16.24.08-02.07.2022222' # 1d trim from yesterday
        # model_name = f'13.42.11-02.08.2022'  # latest trim
        model_name = f'15.59.33-02.08.2022'  # latest xy

        self.model_path = agent_dir + model_name
        if os.path.exists(self.model_path + '_critic'):
            rospy.loginfo(f'Loading model {self.model_path}')
            self.agent.load_checkpoint(self.model_path)
            self.start_time = model_name
            self.last_episode = 474
        else:
            rospy.loginfo('No model found. Training from the beginning...')
            self.start_time = datetime.datetime.now().strftime("%H.%M.%S-%m.%d.%Y")
            self.model_path = agent_dir + self.start_time
            self.last_episode = -1

        # Set up tensorboard logging
        writer_dir = '/tensorboard_logs'
        writer_path = self.path_dir + writer_dir
        self.tf_writer = SummaryWriter(os.path.join(writer_path, self.start_time))

    def train(self):
        """
        Training loop
        """
        np.set_printoptions(precision=2, suppress=True)

        # book-keeping
        evaluations = []
        epoch_rewards = []

        for epoch in range(self.last_episode + 1, self.last_episode + self.train_epoch + 1):
            state = self.env.reset()  # it should reset to initial state here
            # state = self.env.get_observation()
            epoch_reward = 0
            done = False

            # for plots
            actions = np.zeros([self.max_timesteps, 2])
            states = np.zeros([self.max_timesteps, 6])
            t = np.linspace(0,self.max_timesteps,self.max_timesteps).astype(int)

            ts = 0
            while ts < self.max_timesteps:
                # Calculate action
                action = self.agent.select_action(state)
                action += self.expl_noise()  # exploration
                np.clip(action, -1, 1, out=action)

                # for plots
                actions[ts] = action
                states[ts] = state

                # Make action
                # lcg, vbs = action
                # action_6d = np.array([0., 0., 0., 0., lcg, vbs])
                rpm, dr = action
                action_6d = np.array([rpm, rpm, 0., dr, 0., 0.])
                next_state, reward, done = self.env.step(action_6d)

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
            rospy.loginfo('Epoch: {}  Steps: {} Reward: {:.2f}  End state: {}'.format(
                epoch, ts, epoch_reward, state))

            self.tf_writer.add_scalar('reward', epoch_reward, epoch)
            if epoch % 10 == 0:
                self.agent.save_checkpoint(self.model_path)

                # visualize actions and Z
                plot_dir = self.path_dir + '/plots/train/'
                # plot_trim(epoch, plot_dir, t, states, actions)
                plot_xy(epoch, plot_dir, t, states, actions)

    def test(self):
        evaluations = []
        epoch_rewards = []

        test_epochs = 10
        for epoch in range(0, test_epochs):
            # for plots
            actions = np.zeros([self.max_timesteps, 2])
            states = np.zeros([self.max_timesteps, 6])
            t = np.linspace(0,self.max_timesteps,self.max_timesteps).astype(int)

            state = self.env.reset()  # it should reset to initial state here
            # state = self.env.get_observation()
            epoch_reward = 0
            done = False

            ts = 0
            while ts < self.max_timesteps:
                # Calculate action
                action = self.agent.select_action(state)
                action += self.expl_noise()  # exploration
                np.clip(action, -1, 1, out=action)

                # action[0] = (action[0] + 100) / 2
                # action += self.expl_noise()  # exploration
                # np.clip(action, 0, 100, out=action)

                actions[ts] = action
                states[ts] = state

                # Make action
                # lcg, vbs = action
                # action_6d = np.array([0., 0., 0., 0., lcg, vbs])
                rpm, dr = action
                action_6d = np.array([rpm, rpm, 0., dr, 0., 0.])
                next_state, reward, done = self.env.step(action_6d)

                state = next_state
                epoch_reward += reward

                ts += 1
                if done:
                    break

            # After each epoch
            epoch_rewards.append(epoch_reward)
            rospy.loginfo('--------------Epoch: {} Steps: {} Reward: {}'.format(
                epoch, ts, epoch_reward))

            # visualize actions and Z
            plot_dir = self.path_dir + '/plots/test/'
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            # plot_trim(epoch, plot_dir, t, states, actions)
            plot_xy(epoch, plot_dir, t, states, actions)
            plot_traj_2d(epoch, plot_dir, states)

if __name__ == "__main__":
    rospy.init_node("rl_trainer")
    trainer = Trainer(rospy.get_name())
    # trainer.train()
    trainer.test()