cmake_minimum_required(VERSION 3.8)
project(harness_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY urdf rviz launch
  DESTINATION share/${PROJECT_NAME})

ament_package()
