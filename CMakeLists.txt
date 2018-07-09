cmake_minimum_required(VERSION 3.5)
project(trt_pose)

set(CMAKE_CXX_FLAGS -std=c++11)

set(CUDA_NVCC_FLASGS -std=c++11)

#if (CMAKE_BUILD_TYPE == DEBUG)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -G -g)
  #endif()

find_package(CUDA REQUIRED)


include_directories(${CUDA_INCLUDE_DIRS})
add_library(trt_pose SHARED
  src/ParseObjects.cpp
  src/PoseModel.cpp
  src/PafCostGraph.cpp
  src/Munkres.cpp
  src/FindPeaks.cpp
  src/ConnectParts.cpp
)

target_link_libraries(trt_pose ${CUDA_LIBRARIES} nvinfer)
set_target_properties(trt_pose PROPERTIES PUBLIC_HEADER
  "src/PoseModel.hpp;src/Config.hpp;src/tensor.h"
)

install(TARGETS trt_pose
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
  PUBLIC_HEADER DESTINATION include/trt_pose)

if (BUILD_TESTS)
  find_package(GTest REQUIRED)

  add_executable(test_all test/test_all.cpp)
  target_link_libraries(test_all gtest pthread trt_pose)

  add_executable(test_munkres test/test_munkres.cpp)
  target_link_libraries(test_munkres gtest pthread trt_pose)

  add_executable(test_connect_parts test/test_connect_parts.cpp)
  target_link_libraries(test_connect_parts gtest pthread trt_pose)

  add_executable(test_find_peaks test/test_find_peaks.cpp)
  target_link_libraries(test_find_peaks gtest pthread trt_pose)

  add_executable(test_paf_cost_graph test/test_paf_cost_graph.cpp)
  target_link_libraries(test_paf_cost_graph gtest pthread trt_pose)

  add_executable(test_parse_objects test/test_parse_objects.cpp)
  target_link_libraries(test_parse_objects gtest pthread trt_pose)

  find_package(OpenCV REQUIRED)
  add_executable(test_pose_model test/test_pose_model.cpp)
  target_link_libraries(test_pose_model gtest pthread ${OpenCV_LIBS} trt_pose)
endif()
