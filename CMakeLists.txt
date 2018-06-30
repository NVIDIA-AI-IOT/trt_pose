cmake_minimum_required(VERSION 3.5)
project(trt_pose)

set(CMAKE_CXX_STANDARD 11)

add_library(trt_pose SHARED
  src/TensorLayout.cpp
  src/TensorLayout.h
)

if (BUILD_TESTS)
  find_package(GTest REQUIRED)

  add_executable(TestTensorLayout test/TestTensorLayout.cpp)
  target_link_libraries(TestTensorLayout gtest pthread trt_pose)
endif()
