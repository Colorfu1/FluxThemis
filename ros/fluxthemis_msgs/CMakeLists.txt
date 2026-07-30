cmake_minimum_required(VERSION 3.0.2)
project(fluxthemis_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  sensor_msgs
  std_msgs
)

add_service_files(
  FILES
  PredictAction.srv
  ReportEvaluation.srv
)

generate_messages(
  DEPENDENCIES
  sensor_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
  message_runtime
  sensor_msgs
  std_msgs
)
