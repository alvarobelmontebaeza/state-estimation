cmake_minimum_required(VERSION 2.8.3)
project(encoder_localization)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  geometry_msgs
  duckietown_msgs
)

catkin_package()
