cmake_minimum_required(VERSION 3.5)
project(trt_pose)

set(CMAKE_CXX_FLAGS -std=c++11)

set(CUDA_NVCC_FLASGS -std=c++11)

#if (CMAKE_BUILD_TYPE == DEBUG)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -G -g)
  #endif()

find_package(CUDA REQUIRED)

#add_library(trt_pose SHARED
#)

#target_link_libraries(trt_pose)

if (BUILD_TESTS)
  find_package(GTest REQUIRED)

  add_executable(test_all test/test_all.cpp)
  target_link_libraries(test_all gtest pthread)# trt_pose)

  add_executable(test_munkres test/test_munkres.cpp)
  target_link_libraries(test_munkres gtest pthread)# trt_pose)

  add_executable(test_connect_parts test/test_connect_parts.cpp)
  target_link_libraries(test_connect_parts gtest pthread)# trt_pose)

  add_executable(test_find_peaks test/test_find_peaks.cpp)
  target_link_libraries(test_find_peaks gtest pthread)# trt_pose)
endif()
