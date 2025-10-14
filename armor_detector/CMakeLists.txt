cmake_minimum_required(VERSION 3.8)
project(armor_detector)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)

find_library(MVS_LIBRARY
  NAMES MvCameraControl
  PATHS ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

# 如果库没找到，立即报错并停止
if(NOT MVS_LIBRARY)
  message(FATAL_ERROR "Hikvision MVS library (libMvCameraControl.so) not found in ${CMAKE_CURRENT_SOURCE_DIR}/lib. Please check the path.")
endif()

# 打印找到的库路径，方便调试
message(STATUS "Found MVS library: ${MVS_LIBRARY}")



add_executable(armor_detector_node src/armor_detector_node.cpp)

target_include_directories(armor_detector_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${cv_bridge_INCLUDE_DIRS}
  $<INSTALL_INTERFACE:include>)

# 步骤1：只用 ament_target_dependencies 链接 ROS 2 的包
ament_target_dependencies(armor_detector_node
  rclcpp
  sensor_msgs
  image_transport
  cv_bridge
  OpenCV
)

target_link_libraries(armor_detector_node
  ${MVS_LIBRARY}
  ${OpenCV_LIBS}
)


install(TARGETS
  armor_detector_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(
  DIRECTORY lib/
  DESTINATION lib
  FILES_MATCHING PATTERN "*.so*" # 确保所有 .so 文件及其可能的符号链接都被安装
)

install(DIRECTORY
  launch/
  DESTINATION share/${PROJECT_NAME}/
)

install(DIRECTORY
  rviz
  DESTINATION share/${PROJECT_NAME}/
)


install(
      DIRECTORY models/model
      DESTINATION share/${PROJECT_NAME}
    )

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
