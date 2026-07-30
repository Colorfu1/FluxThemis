cmake_minimum_required(VERSION 3.8)
project(fluxthemis_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/PredictAction.srv"
  "srv/ReportEvaluation.srv"
  DEPENDENCIES
  sensor_msgs
  std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
