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
import utils
import torch

# ROS
import rospy
from nav_msgs.msg import Path, Odometry
#from geometry_msgs.msg import PoseStamped, PointStamped

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
                         self.state_feedback_cb)
        rospy.Subscriber(setpoint_topic, Odometry, self.setpoint_cb)
        rospy.Subscriber(enable_topic, Bool, self.enable_cb)

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
        self.current_setpoint = []
        self.state = []
        self.output = []

    def state_feedback_cb(self, odom_msg):
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
        yaw = (1.571-rpy[2])

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
        current_output = np.array(
            [x, y, z, roll, pitch, yaw, u, v, w, p, q, r])

        self.state = current_state
        self.output = current_output

    def setpoint_cb(self, odom_msg):
        """ 
        Subscribe to reference trajectory
        """
        # Insert functon to get the publshed reference trajectory here
        #print('getting setpont')
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
        self.current_setpoint = np.array([x, z, pitch, u, w, q])

    def enable_cb(self):
        pass


class Trainer(ROS_SAM):
    def __init__(self, node_name):
        super(Trainer, self).__init__()

        # Launch parameters
        # self.train_episodes = rospy.get_param("~train_episodes", 100.0)

        # init RL agent
        # agent =
        # agent = 0

        # # Load model weights and metadata if exist
        # agent_dir = os.getcwd()
        # model = f'{agent_dir}/sam_model.mdl'
        # if os.path.exists(model):
        #     rospy.loginfo('Loading model...')
        #     self.start_time, self.last_episode = agent.load_model(model)
        # else:
        #     rospy.loginfo('No model found. Training from the beginning...')
        #     self.start_time = datetime.datetime.now().strftime("%H.%M.%S-%m.%d.%Y")
        #     self.last_episode = -1

        # # Set up tensorboard logging
        # tf_writer = utils.tensorboard.SummaryWriter(
        #     os.path.join('tensorboard_logs', self.start_time))

    def start(self):
        """
        Training loop
        """
        self.rate = rospy.Rate(self.loop_freq)
        while not rospy.is_shutdown():
            self.train_iter()
            self.rate.sleep()

    def train_iter(self):
        pass


if __name__ == "__main__":
    rospy.init_node("rl_trainer")
    trainer = Trainer(rospy.get_name())
    trainer.start()
