#! /usr/bin/env python

# Wrapper node to read control inputs and republish at above 10hz

import rospy
from sam_msgs.msg import ThrusterAngles, PercentStamped
from smarc_msgs.msg import ThrusterRPM
from std_msgs.msg import Bool, Float64, Header


class ActuatorInterface(object):

    def __init__(self, name):
        rl_rpm1_topic = rospy.get_param(
            '~rl_rpm1_topic', default='/sam/ctrl/rl/rpm1')
        rl_rpm2_topic = rospy.get_param(
            '~rl_rpm2_topic', default='/sam/ctrl/rl/rpm2')
        rl_vec_topic = rospy.get_param(
            '~rl_de_dr_topic', default='/sam/ctrl/rl/vec')
        rl_vbs_topic = rospy.get_param(
            '~rl_vbs_topic', default='/sam/ctrl/rl/vbs')
        rl_lcg_topic = rospy.get_param(
            '~rl_lcg_topic', default='/sam/ctrl/rl/lcg')

        enable_topic = rospy.get_param(
            '~rl_enable_topic', default='/sam/ctrl/rl/enable')

        self.loop_freq = rospy.get_param("~loop_freq", default=21)

        self.rpm1_sub = rospy.Subscriber(
            rl_rpm1_topic, ThrusterRPM, self.rpm1_cb)
        self.rpm2_sub = rospy.Subscriber(
            rl_rpm2_topic, ThrusterRPM, self.rpm2_cb)
        self.vec_sub = rospy.Subscriber(
            rl_vec_topic, ThrusterAngles, self.vec_cb)
        self.vbs_sub = rospy.Subscriber(
            rl_vbs_topic, PercentStamped, self.vbs_cb)
        self.lcg_sub = rospy.Subscriber(
            rl_lcg_topic, PercentStamped, self.lcg_cb)

        #rospy.Subscriber(enable_topic, Bool, self.enable_cb)

        self.rpm1_pub = rospy.Publisher(
            "/sam/core/thruster1_cmd", ThrusterRPM, queue_size=10)
        self.rpm2_pub = rospy.Publisher(
            "/sam/core/thruster2_cmd", ThrusterRPM, queue_size=10)
        self.vec_pub = rospy.Publisher(
            "/sam/core/thrust_vector_cmd", ThrusterAngles, queue_size=10)
        self.vbs_pub = rospy.Publisher(
            "/sam/core/vbs_cmd", PercentStamped, queue_size=10)
        self.lcg_pub = rospy.Publisher(
            "/sam/core/lcg_cmd", PercentStamped, queue_size=10)

        self.rate = rospy.Rate(self.loop_freq)

        # initialize actuator commands
        self.lcg = PercentStamped()
        self.vbs = PercentStamped()
        self.rpm1 = ThrusterRPM()
        self.rpm2 = ThrusterRPM()
        self.vec = ThrusterAngles()

        self.enable_flag = True

        while not rospy.is_shutdown():

            if self.enable_flag:

                # publish to actuators
                self.rpm1_pub.publish(self.rpm1)
                self.rpm2_pub.publish(self.rpm2)
                self.vec_pub.publish(self.vec)
                self.vbs_pub.publish(self.vbs)
                self.lcg_pub.publish(self.lcg)

            self.rate.sleep()

    def rpm1_cb(self, rpm):
        self.rpm1.rpm = rpm.rpm

    def rpm2_cb(self, rpm):
        self.rpm2.rpm = rpm.rpm

    def vec_cb(self, vec):
        self.vec.thruster_horizontal_radians = vec.thruster_horizontal_radians
        self.vec.thruster_vertical_radians = vec.thruster_vertical_radians

    def vbs_cb(self, vbs):
        self.vbs.value = vbs.value

    def lcg_cb(self, lcg):
        self.lcg.value = lcg.value

    # Callback function to check for enable flag
    def enable_cb(self, enable_msg):
        #print('Enable:', enable_msg.data)
        if (not enable_msg.data):
            self.enable_flag = False
            rospy.loginfo_throttle(5, 'rl disabled')

        else:
            self.enable_flag = True
            rospy.loginfo_throttle(5, 'rl enabled')


if __name__ == "__main__":
    rospy.init_node("actuator_interface")
    actuator = ActuatorInterface(rospy.get_name())
