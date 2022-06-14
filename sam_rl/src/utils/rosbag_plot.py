#!/usr/bin/env python
from test_plot import plot_traj_2d
from nav_msgs.msg import Odometry
from sam_msgs.msg import ThrusterAngles, PercentStamped
from smarc_msgs.msg import ThrusterRPM
import rosbag
import pandas as pd

import math


def euler_from_quaternion2(pose):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x, y, z, w = pose
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return [roll_x, pitch_y, yaw_z]  # in radians


# The bag file should be in the same directory as your terminal
bag = rosbag.Bag("./xy_rosbag.bag")
topic = ["/sam/dr/odom", "/sam/dr/yaw"]
column_names = ["x", "y", "z", "phi", "theta", "psi"]
df = pd.DataFrame(columns=column_names)

for topic, msg, t in bag.read_messages(topics=topic):
    if topic == "/sam/dr/odom":
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        quat_x = msg.pose.pose.orientation.x
        quat_y = msg.pose.pose.orientation.y
        quat_z = msg.pose.pose.orientation.z
        quat_w = msg.pose.pose.orientation.w
        pose = (quat_x, quat_y, quat_z, quat_w)
        roll, pitch, yaw = euler_from_quaternion2(pose)
    df = df.append(
        {"x": x, "y": y, "z": 0, "phi": roll, "theta": pitch, "psi": yaw},
        ignore_index=True,
    )

data = df.to_numpy()
print(data.shape)
print(data[-1, :])
plot_dir = "/home/gwozniak/catkin_ws/src/smarc_rl_controllers/sam_rl/logs/rosbag_plots/"
plot_traj_2d("rosbag data", 1, plot_dir, 0, data, [])
df.to_csv("out.csv")

# rospy.Subscriber("/sam/core/thrust_vector_cmd", ThrusterAngles, callback_odom)
# rospy.Subscriber("/sam/core/thruster1_cmd", ThrusterRPM, callback_odom)
