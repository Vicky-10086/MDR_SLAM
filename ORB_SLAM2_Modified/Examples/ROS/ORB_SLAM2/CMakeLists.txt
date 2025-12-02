cmake_minimum_required(VERSION 3.0.2)
project(ORB_SLAM2)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
find_package(catkin REQUIRED COMPONENTS cv_bridge image_transport roscpp rospy std_msgs tf message_generation)
add_message_files(
  FILES
  KeyFrameMsg.msg
)
generate_messages(
  DEPENDENCIES
  std_msgs sensor_msgs geometry_msgs
)
find_package(Boost REQUIRED COMPONENTS system thread)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 3.4.5 REQUIRED)
find_package(Pangolin REQUIRED)
catkin_package(CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs cv_bridge tf message_runtime)
set(ORB_SLAM2_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
include_directories(include ${catkin_INCLUDE_DIRS}
  ${ORB_SLAM2_ROOT_DIR}/include
  ${ORB_SLAM2_ROOT_DIR}/Thirdparty/DBoW2
  ${ORB_SLAM2_ROOT_DIR}/Thirdparty/g2o
  /usr/include/eigen3)
set(ORB_LIBS ${ORB_SLAM2_ROOT_DIR}/lib/libORB_SLAM2.so
  ${ORB_SLAM2_ROOT_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
  ${ORB_SLAM2_ROOT_DIR}/Thirdparty/g2o/lib/libg2o.so
  ${Pangolin_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES} "GL")
add_executable(Mono src/ros_mono.cc)
target_link_libraries(Mono ${ORB_LIBS} ${catkin_LIBRARIES})
add_executable(Stereo src/ros_stereo.cc)
target_link_libraries(Stereo ${ORB_LIBS} ${catkin_LIBRARIES})
add_executable(RGBD src/ros_rgbd.cc)
target_link_libraries(RGBD ${ORB_LIBS} ${catkin_LIBRARIES})
add_executable(astra src/astra.cc)
target_link_libraries(astra ${ORB_LIBS} ${catkin_LIBRARIES})
add_executable(indemind src/indemind.cc)
target_link_libraries(indemind ${ORB_LIBS} ${catkin_LIBRARIES})
add_executable(MonoAR src/AR/ros_mono_ar.cc src/AR/ViewerAR.cc)
target_link_libraries(MonoAR ${ORB_LIBS} ${catkin_LIBRARIES})
