cmake_minimum_required(VERSION 2.8.3)
project(fused_localization)

find_package(catkin REQUIRED COMPONENTS
  rospy
  tf
  geometry_msgs
  duckietown_msgs
  sensor_msgs
  message_generation
)

add_service_files(
  FILES
  CalibratePose.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  duckietown_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime
)
