cmake_minimum_required(VERSION 3.8)
project(drone_detector)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(yolo_msgs REQUIRED)

add_executable(drone_detector_node src/drone_detector_node.cpp)
ament_target_dependencies(drone_detector_node
  rclcpp std_msgs yolo_msgs)

install(TARGETS drone_detector_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
