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

import rospy
import numpy as np
import time

from . import utils

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