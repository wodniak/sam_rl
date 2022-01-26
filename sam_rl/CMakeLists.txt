cmake_minimum_required(VERSION 3.0.2)
project(sam_rl)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  roscpp
  rospy
  sam_msgs
  std_msgs
  tf
  actionlib
  smarc_msgs
)

catkin_package(
)

include_directories(
  #include
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  src/mpc_action.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

 install(PROGRAMS
   scripts/mpc_action.py 
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )

catkin_install_python(PROGRAMS 
  scripts/mpc_action.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  PATTERN ".svn" EXCLUDE
)
