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

from cmath import sqrt
import os
import datetime
import numpy as np
import utils
import torch
import time

from torch.utils.tensorboard import SummaryWriter
from agents.agent_ddpg import DDPGAgent, ReplayBuffer, OUActionNoise

# ROS
import rospy
from nav_msgs.msg import Path, Odometry

# SMARC
from smarc_msgs.msg import ThrusterRPM
from sam_msgs.msg import ThrusterAngles, PercentStamped
from std_msgs.msg import Float64, Header, Bool


class ROS_SAM(object):
    def __init__(self):
        # Launch parameters
        self.xy_tolerance = rospy.get_param("~xy_tolerance", default=1.0)
        self.depth_tolerance = rospy.get_param("~depth_tolerance", default=0.5)
        self.loop_freq = rospy.get_param("~loop_freq", default=11)

        # Topics for feedback and actuators
        state_feedback_topic = rospy.get_param(
            "~state_feedback_topic", default="/sam/sim/odom")
        setpoint_topic = rospy.get_param(
            "~setpoint_topic", default="/sam/ctrl/rl/setpoint")
        vbs_topic = rospy.get_param("~vbs_topic", default="/sam/core/vbs_cmd")
        lcg_topic = rospy.get_param("~lcg_topic", default="/sam/core/lcg_cmd")
        rpm1_topic = rospy.get_param(
            "~rpm1_topic", default="/sam/core/thruster1_cmd")
        rpm2_topic = rospy.get_param(
            "~rpm2_topic", default="/sam/core/thruster2_cmd")
        thrust_vector_cmd_topic = rospy.get_param(
            "~thrust_vector_cmd_topic", default="/sam/core/thrust_vector_cmd")
        enable_topic = rospy.get_param(
            "~enable_topic", default="/sam/ctrl/rl/enable")

        # Subscribers to state feedback, setpoints and enable flags
        rospy.Subscriber(state_feedback_topic, Odometry,
                         self._state_feedback_cb)
        rospy.Subscriber(setpoint_topic, Odometry, self._setpoint_cb)
        rospy.Subscriber(enable_topic, Bool, self._enable_cb)

        # Publishers to actuators
        queue_size = 1
        self.rpm1_pub = rospy.Publisher(
            rpm1_topic, ThrusterRPM, queue_size=queue_size)
        self.rpm2_pub = rospy.Publisher(
            rpm2_topic, ThrusterRPM, queue_size=queue_size)
        self.vec_pub = rospy.Publisher(
            thrust_vector_cmd_topic, ThrusterAngles, queue_size=queue_size)
        self.vbs_pub = rospy.Publisher(
            vbs_topic, PercentStamped, queue_size=queue_size)
        self.lcg_pub = rospy.Publisher(
            lcg_topic, PercentStamped, queue_size=queue_size)

        # Variables
        # self.current_setpoint = np.array(
        #     [0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.current_setpoint = np.array(
            [5.0, 0.0])
        self.state = []
        self.state_timestamp = 0
        self.output = []

    def _state_feedback_cb(self, odom_msg):
        """
        Subscribe to state feedback
        """
        x = odom_msg.pose.pose.position.y
        y = odom_msg.pose.pose.position.x
        z = -odom_msg.pose.pose.position.z
        eta0 = odom_msg.pose.pose.orientation.w
        eps1 = odom_msg.pose.pose.orientation.x
        eps2 = odom_msg.pose.pose.orientation.y
        eps3 = odom_msg.pose.pose.orientation.z

        rpy = utils.euler_from_quaternion([eps1, eps2, eps3, eta0])
        roll = rpy[0]
        pitch = rpy[1]
        yaw = (1.571-rpy[2]) # ENU to NED

        u = odom_msg.twist.twist.linear.x
        v = odom_msg.twist.twist.linear.y
        w = odom_msg.twist.twist.linear.z
        p = odom_msg.twist.twist.angular.x
        q = odom_msg.twist.twist.angular.y
        r = odom_msg.twist.twist.angular.z

        # state : (12 x 1)
        # position (3), rotation (3), linear velocity (3), angular velocity (3)
        # with euler angles
        # current_state = np.array([x, y, z, roll, pitch, yaw, u, v, w, p, q, r])
        current_state = np.array([z, w])
        self.state = current_state
        self.state_timestamp = odom_msg.header.stamp

    def _setpoint_cb(self, odom_msg):
        """
        Subscribe to reference trajectory
        """
        # Insert functon to get the publshed reference trajectory here
        # print('getting setpont')
        x = odom_msg.pose.pose.position.x
        y = odom_msg.pose.pose.position.y
        z = odom_msg.pose.pose.position.z
        eta0 = odom_msg.pose.pose.orientation.w
        eps1 = odom_msg.pose.pose.orientation.x
        eps2 = odom_msg.pose.pose.orientation.y
        eps3 = odom_msg.pose.pose.orientation.z

        # converting quaternion to euler
        rpy = utils.euler_from_quaternion([eps1, eps2, eps3, eta0])
        roll = rpy[0]
        pitch = rpy[1]
        yaw = rpy[2]

        u = odom_msg.twist.twist.linear.x
        v = odom_msg.twist.twist.linear.y
        w = odom_msg.twist.twist.linear.z
        p = odom_msg.twist.twist.angular.x
        q = odom_msg.twist.twist.angular.y
        r = odom_msg.twist.twist.angular.z

        state = np.array([x, y, z, roll, pitch, yaw, u, v, w, p, q, r])
        self.current_setpoint = state

    def _enable_cb(self):
        pass

    def publish_actions(self, action):
        # lcg = PercentStamped()
        vbs = PercentStamped()
        # rpm1 = ThrusterRPM()
        # rpm2 = ThrusterRPM()
        # vec = ThrusterAngles()

        vbs.value = action[0]
        # lcg.value = action[1]
        # self.rpm1_pub.publish(rpm1)
        # self.rpm2_pub.publish(rpm2)
        # self.vec_pub.publish(vec)
        self.vbs_pub.publish(vbs)
        # self.lcg_pub.publish(lcg)


class SAMEnv(ROS_SAM):
    def __init__(self):
        super(SAMEnv, self).__init__()

        self.state_dim = 2
        self.action_dim = 1  # LCG, VBS
        self.max_action = np.array([100])  # max LCG = 1, max VBS = 1

        self.last_state_timestamp = self.state_timestamp

    def get_observation(self):
        """Return state with different timestamp then before"""

        time.sleep(0.011)
        self.last_state_timestamp = self.state_timestamp
        return self.state

    def step(self, action):
        self.publish_actions(action)

        # probably after a short wait
        next_state = self.get_observation()
        reward = self._calculate_reward(next_state, self.current_setpoint)
        done = False
        return next_state, reward, done

    def reset(self):
        # reset state to initial
        vbs = PercentStamped()
        self.vbs_pub.publish(vbs)
        time.sleep(10)
        return self.get_observation()

    def _calculate_reward(self, state, target):
        # penalize on depth and elevation angle
        # add control penalty
        error = np.power(state[0] - target[0], 2)
        return -error

    def _is_done(self, state, target):
        eps = 1e-3
        return np.isclose(state[2], target[2], rtol=eps)


class Trainer(object):
    def __init__(self, node_name):
        # Parameters
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        # Init environment
        self.env = SAMEnv()

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
        self.max_timesteps = 6000  # per epoch

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
                action += self.expl_noise()  # exploration

                np.clip(action, 0, self.env.max_action[0], out=action)

                # Make action
                next_state, reward, done = self.env.step(action)
                state = next_state
                epoch_reward += reward

                ts += 1
                if done:
                    break

            # After each epoch
            epoch_rewards.append(epoch_reward)
            rospy.loginfo('Epoch: {} Steps: {} Reward: {}'.format(
                epoch, ts, epoch_reward))


if __name__ == "__main__":
    rospy.init_node("rl_trainer")
    trainer = Trainer(rospy.get_name())
    trainer.train()
    # trainer.test()