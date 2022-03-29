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

# ROS
import rospy
from nav_msgs.msg import Path, Odometry

# SMARC
from smarc_msgs.msg import ThrusterRPM
from sam_msgs.msg import ThrusterAngles, PercentStamped
from std_msgs.msg import Float64, Header, Bool
from . import utils


class SamRosInterface(object):
    """Use stonefish SAM simulator"""

    def __init__(self):
        # Launch parameters
        self.xy_tolerance = rospy.get_param("~xy_tolerance", default=1.0)
        self.depth_tolerance = rospy.get_param("~depth_tolerance", default=0.5)
        self.loop_freq = rospy.get_param("~loop_freq", default=11)

        # Topics for feedback and actuators
        state_feedback_topic = rospy.get_param(
            "~state_feedback_topic", default="/sam/sim/odom"
        )
        setpoint_topic = rospy.get_param(
            "~setpoint_topic", default="/sam/ctrl/rl/setpoint"
        )
        vbs_topic = rospy.get_param("~vbs_topic", default="/sam/core/vbs_cmd")
        lcg_topic = rospy.get_param("~lcg_topic", default="/sam/core/lcg_cmd")
        rpm1_topic = rospy.get_param("~rpm1_topic", default="/sam/core/thruster1_cmd")
        rpm2_topic = rospy.get_param("~rpm2_topic", default="/sam/core/thruster2_cmd")
        thrust_vector_cmd_topic = rospy.get_param(
            "~thrust_vector_cmd_topic", default="/sam/core/thrust_vector_cmd"
        )
        enable_topic = rospy.get_param("~enable_topic", default="/sam/ctrl/rl/enable")

        # Subscribers to state feedback, setpoints and enable flags
        rospy.Subscriber(state_feedback_topic, Odometry, self._state_feedback_cb)
        rospy.Subscriber(setpoint_topic, Odometry, self._setpoint_cb)
        rospy.Subscriber(enable_topic, Bool, self._enable_cb)

        # Publishers to actuators
        queue_size = 1
        self.rpm1_pub = rospy.Publisher(rpm1_topic, ThrusterRPM, queue_size=queue_size)
        self.rpm2_pub = rospy.Publisher(rpm2_topic, ThrusterRPM, queue_size=queue_size)
        self.vec_pub = rospy.Publisher(
            thrust_vector_cmd_topic, ThrusterAngles, queue_size=queue_size
        )
        self.vbs_pub = rospy.Publisher(vbs_topic, PercentStamped, queue_size=queue_size)
        self.lcg_pub = rospy.Publisher(lcg_topic, PercentStamped, queue_size=queue_size)

        # Variables
        self.full_state_dim = 12
        self.full_state = np.zeros(self.full_state_dim)
        self.current_setpoint = np.zeros(self.full_state_dim)

        self.state_timestamp = 0
        self.enable = True

    def _state_feedback_cb(self, odom_msg):
        """
        Subscribe to state feedback
        """
        x = odom_msg.pose.pose.position.y
        y = odom_msg.pose.pose.position.x
        z = odom_msg.pose.pose.position.z
        eta0 = odom_msg.pose.pose.orientation.w
        eps1 = odom_msg.pose.pose.orientation.x
        eps2 = odom_msg.pose.pose.orientation.y
        eps3 = odom_msg.pose.pose.orientation.z

        rpy = utils.euler_from_quaternion([eps1, eps2, eps3, eta0])
        roll = rpy[0]
        pitch = -rpy[1]
        yaw = 1.571 - rpy[2]  # ENU to NED

        u = odom_msg.twist.twist.linear.x
        v = odom_msg.twist.twist.linear.y
        w = odom_msg.twist.twist.linear.z
        p = odom_msg.twist.twist.angular.x
        q = odom_msg.twist.twist.angular.y
        r = odom_msg.twist.twist.angular.z

        # state : (12 x 1)
        # position (3), rotation (3), linear velocity (3), angular velocity (3)
        # with euler angles
        current_state = np.array([x, y, z, roll, pitch, yaw, u, v, w, p, q, r])
        self.full_state = current_state
        self.state_timestamp = odom_msg.header.stamp

    def _setpoint_cb(self, odom_msg):
        """
        Subscribe to reference trajectory
        """
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
        pitch = -rpy[1]
        yaw = rpy[2]

        u = odom_msg.twist.twist.linear.x
        v = odom_msg.twist.twist.linear.y
        w = odom_msg.twist.twist.linear.z
        p = odom_msg.twist.twist.angular.x
        q = odom_msg.twist.twist.angular.y
        r = odom_msg.twist.twist.angular.z

        state = np.array([x, y, z, roll, pitch, yaw, u, v, w, p, q, r])
        print(f"Setting new target to {state}")
        self.current_setpoint = state

    def _enable_cb(self, enable_msg):
        self.enable = enable_msg.data
        print(f"Setting Enable to {self.enable}")

    def publish_actions(self, action):
        """
        :param action (6,) for (rpm, rpm, de, dr, lcg, vbs)
        """
        if not self.enable:
            return

        rpm = ThrusterRPM()
        vec = ThrusterAngles()
        vbs = PercentStamped()
        lcg = PercentStamped()

        rpm.rpm = action[0].astype(int)
        vec.thruster_vertical_radians = action[2]
        vec.thruster_horizontal_radians = action[3]
        lcg.value = action[4]
        vbs.value = action[5]

        self.rpm1_pub.publish(rpm)
        self.rpm2_pub.publish(rpm)
        self.vec_pub.publish(vec)
        self.lcg_pub.publish(lcg)
        self.vbs_pub.publish(vbs)


class SAMEnv(SamRosInterface):
    """
    SAM in stonefish
    Underlying ROS interface operates on 12d full state
    `env_obs_states` and `env_actions` specify what variables are used by the policy.
    """

    def __init__(self, env_obs_states, env_actions):
        super(SAMEnv, self).__init__()

        self.env_obs_states = env_obs_states  # defined in params
        self.env_actions = env_actions
        self.key_to_state_map = {
            "x": 0,
            "y": 1,
            "z": 2,
            "phi": 3,
            "theta": 4,
            "psi": 5,
            "u": 6,
            "v": 7,
            "w": 8,
            "p": 9,
            "q": 10,
            "r": 11,
        }

        self.full_state_dim = 12
        self.full_action_dim = 5  # RPM, DE, DR, LCG, VBS
        self.full_action_scale_norm = {
            "rpm": 1000,
            "de": 0.1,
            "dr": 0.1,
            "lcg": 100,
            "vbs": 100,
        }  # NOTE keys are the same as in `env_actions`

        self.last_state_timestamp = self.state_timestamp

        input_action_size = len(env_actions)
        self.prev_action = np.zeros(input_action_size)

        print(
            f"Running with Env Stonefish:\n\
            Observed states:{env_obs_states.keys()}\n\
            Actions:{env_actions.keys()}\n\
            Action scale:{self.full_action_scale_norm}"
        )

    def _make_action_6d(self, input_action) -> np.ndarray:
        action_6d = np.zeros(6)
        for i, key in enumerate(self.env_actions):
            pos = self.env_actions[key]
            value = input_action[i]

            # scale the value
            scale = self.full_action_scale_norm[key]
            if key == "lcg" or key == "vbs":
                scaled_value = (value + 1) / 2 * scale  # norm to 0-100
            else:
                scaled_value = value * scale  # norm to eg. +-1000

            action_6d[pos] = scaled_value
            if key == "rpm":
                # rpm is the same for both propellers
                action_6d[0] = scaled_value

        return action_6d

    def _is_done(self, state, target):
        eps = 1e-3
        return np.isclose(state[2], target[2], rtol=eps)

    def _get_obs(self):
        """Return state with different timestamp then before"""
        time.sleep(0.011)
        state_12d = self.full_state
        self.last_state_timestamp = self.state_timestamp

        obs = []
        for key in self.key_to_state_map.keys():
            if key in self.env_obs_states:
                state_pos = self.key_to_state_map[key]
                obs.append(state_12d[state_pos])

        return np.array(obs)

    def get_current_setpoint(self):
        """Return setpoint received through ROS"""
        time.sleep(0.011)
        state_12d = self.current_setpoint
        self.last_state_timestamp = self.state_timestamp

        obs = []
        for key in self.key_to_state_map.keys():
            if key in self.env_obs_states:
                state_pos = self.key_to_state_map[key]
                obs.append(state_12d[state_pos])

        return np.array(obs)

    def step(self, action):
        """
        :param action: actions given by the policy network
        """
        action_6d = self._make_action_6d(action)
        self.publish_actions(action_6d)

        # probably after a short wait
        observed_state = self._get_obs()
        reward, reward_info = self._calculate_reward(observed_state, action)
        done = False
        self.prev_action = action

        info_state = {
            "xyz": self.full_state[0:3].round(2).tolist(),
            "rpy": self.full_state[3:6].round(2).tolist(),
            "uvw": self.full_state[6:9].round(2).tolist(),
            "pqr": self.full_state[9:12].round(2).tolist(),
        }
        action_6d = action_6d.round(2).tolist()
        info_actions = {
            "rpm": action_6d[1],
            "de": action_6d[2],
            "dr": action_6d[3],
            "lcg": action_6d[4],
            "vbs": action_6d[5],
        }
        info = {"rewards": reward_info, "state": info_state, "actions": info_actions}

        return observed_state, reward, done, info

    def reset(self):
        """
        Make it go to the surface by changing buyoancy
        """
        # reset state to initial
        vbs = PercentStamped()
        self.vbs_pub.publish(vbs)
        # time.sleep(10)
        observed_state = self._get_obs()
        return observed_state

    def _calculate_reward(self, state, action):
        Q = np.diag(
            # [0.0, 0.1, 0.3, 0.0, 0.3, 0.0],
            [0.0, 0.1, 0.3, 0.0, 0.3, 0.0, 0.0, 0.1, 0.3, 0.0, 0.3, 0.0]
        )  # z, pitch, v, q
        R = np.diag([0.03, 0.03])  # weights on controls
        R_r = np.diag([0.3, 0.3])  # weights on rates

        a_diff = action - self.prev_action

        e_s = np.linalg.norm(state * Q * state)  # error on state, setpoint is all 0's
        e_a = np.linalg.norm(action * R * action)  # error on actions
        e_r = np.linalg.norm(a_diff * R_r * a_diff)  # error on action rates
        e = e_s + e_a + e_r
        error = np.maximum(0, 1.0 - e_s) - e_a - e_r

        e_info = {
            "e_total": round(error, 3),
            "e_state": round(-e_s, 3),
            "e_action": round(-e_a, 3),
            "e_action_rate": round(-e_r, 3),
        }
        return error, e_info
