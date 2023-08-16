cmake_minimum_required(VERSION 2.8.3)
project(pv_lio)

#SET(CMAKE_BUILD_TYPE "Debug")
#set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -pthread -lstdc++fs")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread")

ADD_COMPILE_OPTIONS(-std=c++17 )
ADD_COMPILE_OPTIONS(-std=c++17 )
set(CMAKE_CXX_FLAGS "-std=c++17 -O3")

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions" )
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread -std=c++0x -std=c++17 -lboost_system -fexceptions -lstdc++fs -O3 -march=native -fno-math-errno -ffast-math")

find_package(Boost REQUIRED COMPONENTS timer)
# For mvs dump
FIND_PACKAGE(Boost REQUIRED COMPONENTS filesystem iostreams program_options system serialization)
if(Boost_FOUND)
	INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
	LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})
endif()

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

find_package(PythonLibs REQUIRED)
find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  pcl_ros
  tf
  livox_ros_driver
  message_generation
  eigen_conversions
  image_transport	
  cv_bridge
)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)
#find_package(GTSAM REQUIRED QUIET)

message(Eigen: ${EIGEN3_INCLUDE_DIR})

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  include)

#add_message_files(
#  FILES
#  Pose6D.msg
#)
add_service_files(
  FILES
  SetFloat.srv
)

generate_messages(
 DEPENDENCIES
 geometry_msgs
 std_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime
  DEPENDS EIGEN3 PCL
  INCLUDE_DIRS
)

add_executable(pv_lio_node src/voxelMapping.cpp src/preprocess.cpp src/calibration_data.hpp src/openmvs_utils.hpp src/r3live_coloring.hpp src/services.hpp src/loguru.cpp src/loguru.hpp src/common.hpp)
target_link_libraries(pv_lio_node ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES} stdc++fs ${Boost_LIBRARIES} ${Boost_FILESYSTEM_LIBRARY} ${Boost_SERIALIZATION_LIBRARY})
target_include_directories(pv_lio_node PRIVATE ${PYTHON_INCLUDE_DIRS})

add_executable(
  segment_pc
  src/segment_pc/main.cpp
  src/loguru.cpp src/loguru.hpp
  src/calibration_data.hpp
  src/r3live_coloring.hpp
)

target_link_libraries(
  segment_pc
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${PYTHON_LIBRARIES}
)

target_include_directories(
  segment_pc
  PRIVATE ${PYTHON_INCLUDE_DIRS}
)

# add_executable(pandar_to_velodyne src/converter/pandar_to_velodyne.cpp)
# target_link_libraries(pandar_to_velodyne ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES})
# target_include_directories(pandar_to_velodyne PRIVATE ${PYTHON_INCLUDE_DIRS})

# add_executable(rs_to_velodyne src/converter/rs_to_velodyne.cpp)
# target_link_libraries(rs_to_velodyne ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES})
# target_include_directories(rs_to_velodyne PRIVATE ${PYTHON_INCLUDE_DIRS})
