cmake_minimum_required(VERSION 2.8.3)
project(at_localization)

find_package(catkin REQUIRED COMPONENTS
  rospy
  tf
  geometry_msgs
  duckietown_msgs
  sensor_msgs
  cv_bridge
)

catkin_package()
