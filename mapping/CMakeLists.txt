cmake_minimum_required(VERSION 2.8.3)
project(pointcloud_mapping)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    sensor_msgs
    geometry_msgs
    tf
    message_filters
    cv_bridge
    image_transport
    pcl_ros
    tf2_ros
    tf_conversions
)

find_package(PCL 1.8 REQUIRED)
find_package(OpenCV REQUIRED)


catkin_package(
  LIBRARIES
  CATKIN_DEPENDS
    roscpp
    pcl_ros
  DEPENDS PCL OpenCV
)



include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)


add_executable(mapping src/mapping.cpp)


target_link_libraries(mapping
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
)
