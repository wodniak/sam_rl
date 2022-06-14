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

from sam_msgs.msg import ThrusterAngles, PercentStamped
from smarc_msgs.msg import ThrusterRPM
from std_msgs.msg import Float64, Header
from std_srvs.srv import SetBool
from ddynamic_reconfigure_python.ddynamic_reconfigure import DDynamicReconfigure


class ToggleController(object):
    """a class to define a service client to toggle controllers"""

    def toggle(self, enable_):
        # function that toggles the service, that can be called from the code
        ret = self.toggle_ctrl_service(enable_)
        if ret.success:
            rospy.loginfo_throttle_identical(5, "Controller toggled")

    def __init__(self, service_name_, enable_):
        rospy.wait_for_service(service_name_)
        try:
            self.toggle_ctrl_service = rospy.ServiceProxy(service_name_, SetBool)
            self.toggle(enable_)

        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)


class ReconfigServer(object):
    def __init__(self, configobject):
        """
        Read the defaults from the config object
        so that we dont change the launch-defaults right away
        with different reconfig defaults
        We then just put whatever reconfig we get into the BB with the same key
        """
        # DynamicDynamicReConfig
        # because the .cfg way of doing this is pain
        self.configobject = configobject  # store a copy of the reference to the object
        self.ddrc = DDynamicReconfigure("actions_reconfig")
        # name, description, default value, min, max, edit_method
        self.ddrc.add_variable(
            "desired_depth", "Depth Setpoint", configobject.desired_depth, 0.0, 10.0
        )

        self.ddrc.add_variable(
            "desired_pitch", "Pitch Setpoint", configobject.desired_pitch, -2.0, 2.0
        )

        self.ddrc.add_variable(
            "rpm_value", "RPM Value", configobject.rpm_value, -2000, 2000
        )

        self.ddrc.add_variable(
            "elevator_angle", "Elevator Angle", configobject.elevator_angle, -0.15, 0.15
        )

        self.ddrc.add_variable(
            "rudder_angle", "Rudder Angle", configobject.rudder_angle, -0.15, 0.15
        )

        self.ddrc.add_variable(
            "flip_rate", "Flip Rate", configobject.flip_rate, 0.0, 4.0
        )

        self.ddrc.add_variable(
            "thrust_ratio", "Thrust Ratio", configobject.thrust_ratio, 0.0, 1.0
        )

        self.ddrc.add_variable(
            "left_turn", "Left Turn", configobject.left_turn, False, True
        )

        self.ddrc.add_variable(
            "upwards_turn", "Upwards Turn", configobject.left_turn, False, True
        )

        self.ddrc.add_variable(
            "control_trim_flag",
            "Control Trim while turboturning",
            configobject.control_trim_flag,
            False,
            True,
        )

        self.ddrc.add_variable(
            "control_speed_flag",
            "Control Speed and Trim",
            configobject.control_speed_flag,
            False,
            True,
        )

        rospy.loginfo(
            "Started dynamic reconfig server with keys:{}".format(
                self.ddrc.get_variable_names()
            )
        )

        # this should be the last thing in this init
        self.ddrc.start(self.reconfig_cb)

    def reconfig_cb(self, config, level):
        for key in self.ddrc.get_variable_names():
            new_value = config.get(key)

            rospy.loginfo("New value for:{} set to:{} )".format(key, new_value))

            if key == "desired_depth":
                self.configobject.desired_depth = new_value

            if key == "desired_pitch":
                self.configobject.desired_pitch = new_value

            if key == "rpm_value":
                self.configobject.rpm_value = new_value

            if key == "elevator_angle":
                self.configobject.elevator_angle = new_value

            if key == "rudder_angle":
                self.configobject.rudder_angle = new_value

            if key == "flip_rate":
                self.configobject.flip_rate = new_value

            if key == "thrust_ratio":
                self.configobject.thrust_ratio = new_value

            if key == "left_turn":
                self.configobject.left_turn = new_value

            if key == "upwards_turn":
                self.configobject.left_turn = new_value

            if key == "control_trim_flag":
                self.configobject.control_trim_flag = new_value

            if key == "control_speed_flag":
                self.configobject.control_speed_flag = new_value

        return config


class InvertedPendulum(object):
    """blabla"""

    def pitch_feedback_cb(self, pitch_feedback):
        """blabla"""
        self.pitch_feedback = pitch_feedback.data

    def turbo_turn(self):
        """blabla"""
        # disengage controllers
        self.toggle_speed_ctrl.toggle(False)
        self.toggle_roll_ctrl.toggle(False)

        # set vbs and lcg to constant values (e.g. 30,0)
        # self.lcg.value =  0.
        # self.vbs.value = 30.
        # self.vbs_pub.publish(self.vbs)
        # self.lcg_pub.publish(self.lcg)

        if self.left_turn:
            self.rudder_angle = -self.rudder_angle

        if not self.upwards_turn:
            self.elevator_angle = -self.elevator_angle

        thrust_rate = 11.0
        rate = rospy.Rate(thrust_rate)

        rpm1 = ThrusterRPM()
        rpm2 = ThrusterRPM()

        self.vec_pub.publish(self.elevator_angle, self.rudder_angle, Header())
        loop_time = 0.0
        while (
            not rospy.is_shutdown() and loop_time < self.thrust_ratio / self.flip_rate
        ):
            rpm1.rpm = self.rpm_value
            rpm2.rpm = self.rpm_value
            self.rpm1_pub.publish(rpm1)
            self.rpm2_pub.publish(rpm2)
            loop_time += 1.0 / thrust_rate
            rate.sleep()

        self.vec_pub.publish(-self.elevator_angle, -self.rudder_angle, Header())
        loop_time = 0.0
        while (
            not rospy.is_shutdown()
            and loop_time < (1 - self.thrust_ratio) / self.flip_rate
        ):
            rpm1.rpm = -self.rpm_value
            rpm2.rpm = -self.rpm_value
            self.rpm1_pub.publish(rpm1)
            self.rpm2_pub.publish(rpm2)
            loop_time += 1.0 / thrust_rate
            rate.sleep()
        rospy.loginfo_throttle(5, "Turbo Turning!!")

    def control_trim(self):
        # send setpoints to trim and depth controllers
        self.depth_setpoint.data = self.desired_depth
        self.toggle_depth_ctrl.toggle(True)
        self.depth_pub.publish(self.depth_setpoint)

        self.pitch_setpoint.data = self.desired_pitch
        self.toggle_dpitch_ctrl.toggle(False)
        self.toggle_pitch_ctrl.toggle(True)
        self.pitch_pub.publish(self.pitch_setpoint)
        rospy.loginfo_throttle(5, "Controlling trim!")

    def control_trim_speed(self):
        self.depth_setpoint.data = self.desired_depth
        self.toggle_depth_ctrl.toggle(True)
        self.depth_pub.publish(self.depth_setpoint)

        self.pitch_setpoint.data = self.desired_pitch
        self.toggle_pitch_ctrl.toggle(True)
        self.toggle_dpitch_ctrl.toggle(True)
        self.pitch_pub.publish(self.pitch_setpoint)

        self.roll_setpoint.data = 0.0
        self.toggle_roll_ctrl.toggle(True)
        self.roll_pub.publish(self.roll_setpoint)

        self.speed_setpoint.data = 0.0
        self.toggle_speed_ctrl.toggle(True)
        self.depth_pub.publish(self.speed_setpoint)
        rospy.loginfo_throttle(5, "Engaging controllers!")

    def __init__(self, name):
        self.rpm_value = rospy.get_param("~rpm", 500)
        toggle_depth_ctrl_service = rospy.get_param(
            "~toggle_vbs_ctrl_service", "/sam/ctrl/toggle_vbs_ctrl"
        )
        toggle_pitch_ctrl_service = rospy.get_param(
            "~toggle_pitch_ctrl_service", "/sam/ctrl/toggle_pitch_ctrl"
        )
        toggle_dpitch_ctrl_service = rospy.get_param(
            "~toggle_dpitch_ctrl_service", "/sam/ctrl/toggle_dyn_pitch_ctrl"
        )
        toggle_speed_ctrl_service = rospy.get_param(
            "~toggle_speed_ctrl_service", "/sam/ctrl/toggle_speed_ctrl"
        )
        toggle_roll_ctrl_service = rospy.get_param(
            "~toggle_roll_ctrl_service", "/sam/ctrl/toggle_roll_ctrl"
        )
        self.loop_freq = rospy.get_param("~loop_freq", 21)
        self.flip_rate = rospy.get_param("~flip_rate", 0.5)
        self.thrust_ratio = rospy.get_param("~thrust_ratio", 0.6)  # 0.6
        self.upwards_turn = rospy.get_param("~upwards_turn", True)
        self.left_turn = rospy.get_param("~left_turn", True)

        self.rpm1_pub = rospy.Publisher(
            "/sam/core/thruster1_cmd", ThrusterRPM, queue_size=10
        )
        self.rpm2_pub = rospy.Publisher(
            "/sam/core/thruster2_cmd", ThrusterRPM, queue_size=10
        )
        self.vec_pub = rospy.Publisher(
            "/sam/core/thrust_vector_cmd", ThrusterAngles, queue_size=10
        )
        self.vbs_pub = rospy.Publisher(
            "/sam/core/vbs_cmd", PercentStamped, queue_size=10
        )
        self.lcg_pub = rospy.Publisher(
            "/sam/core/lcg_cmd", PercentStamped, queue_size=10
        )

        self.depth_setpoint = Float64()
        self.pitch_setpoint = Float64()
        self.roll_setpoint = Float64()
        self.speed_setpoint = Float64()
        self.pitch_pub = rospy.Publisher(
            "/sam/ctrl/pitch_setpoint", Float64, queue_size=10
        )
        self.depth_pub = rospy.Publisher(
            "/sam/ctrl/depth_setpoint", Float64, queue_size=10
        )
        self.speed_pub = rospy.Publisher(
            "/sam/ctrl/speed_setpoint", Float64, queue_size=10
        )
        self.roll_pub = rospy.Publisher(
            "/sam/ctrl/roll_setpoint", Float64, queue_size=10
        )

        self.pitch_feedback = 0.0
        self.pitch_feedback_sub = rospy.Subscriber(
            "/sam/ctrl/pitch_feedback", Float64, self.pitch_feedback_cb
        )

        self.rate = rospy.Rate(self.loop_freq)

        # initialize actuator commands
        self.lcg = PercentStamped()
        self.vbs = PercentStamped()
        self.rpm = ThrusterRPM()
        self.vec = ThrusterAngles()

        upward_pendulum = True

        if upward_pendulum:
            self.elevator_angle = 0.15
            self.rudder_angle = 0.0
            self.rpm_value = 1500
            self.left_turn = True
            self.upwards_turn = True
            # self.flip_rate = 0.5 #causes errors
            self.thrust_ratio = 0.4
            self.control_trim_flag = False  # False
            self.control_speed_flag = False  # False
            self.desired_pitch = -1.5
            self.desired_depth = 4.0
        else:
            # Downward pendulum, to be checked
            self.elevator_angle = -0.15
            self.rudder_angle = 0.0
            self.rpm_value = -1500
            self.left_turn = False
            self.upwards_turn = True
            # self.flip_rate = 0.5 # causes errors
            self.thrust_ratio = 0.6
            self.control_trim_flag = False  # False
            self.control_speed_flag = False  # False
            self.desired_pitch = 1.5
            self.desired_depth = 4.0

        """self.elevator_angle = 0.15
        self.rudder_angle = 0.0

        self.control_trim_flag = False
        self.control_speed_flag = False
        self.desired_pitch = -1.5
        self.desired_depth = 4.0"""
        self.toggle_depth_ctrl = ToggleController(toggle_depth_ctrl_service, False)
        self.toggle_pitch_ctrl = ToggleController(toggle_pitch_ctrl_service, False)
        self.toggle_dpitch_ctrl = ToggleController(toggle_dpitch_ctrl_service, False)
        self.toggle_speed_ctrl = ToggleController(toggle_speed_ctrl_service, False)
        self.toggle_roll_ctrl = ToggleController(toggle_roll_ctrl_service, False)

        reconfig = ReconfigServer(self)


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

    print("Loading inverted pendulum")
    periscope = InvertedPendulum(rospy.get_name())
    use_pendulum = False

    episode = 0
    obs = env.reset()
    while True:
        print("New Episode")

        # Reset env
        setpoint = env.get_current_setpoint()

        # for plots
        ep_actions = np.zeros([max_timesteps, 5])
        ep_states = np.zeros([max_timesteps, 12])
        ep_t = np.linspace(0, max_timesteps, max_timesteps).astype(int)

        ep_reward = 0
        for ts in range(max_timesteps):
            if use_pendulum:
                for _ in range(5):
                    periscope.turbo_turn()
                use_pendulum = False
                print("CONTROLLER OFF")
            obs -= setpoint  # will follow setpoint
            action, _states = model.predict(obs)

            obs, rewards, dones, info = env.step(action)
            end_state = obs
            ep_reward += rewards

            ep_actions[ts] = [*info["actions"].values()]  # save
            ep_states[ts] = list(itertools.chain(*info["state"].values()))  # save

            # print(
            #     "[{}] {}\n{}\n{}\n{}\n{}\n".format(
            #         ts, setpoint, info["state"], info["actions"], info["rewards"], obs
            #     )
            # )

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
