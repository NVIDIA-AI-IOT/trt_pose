cmake_minimum_required(VERSION 3.5)
project(trt_pose)

set(CMAKE_CXX_FLAGS -std=c++11)
set(CUDA_NVCC_FLASGS -std=c++11)

find_package(CUDA REQUIRED)

#add_library(trt_pose SHARED
#src/TensorLayout.cpp
#src/TensorLayout.h
#)

if (BUILD_TESTS)
  find_package(GTest REQUIRED)

  cuda_add_executable(TestCudaFunction test/TestCudaFunction.cu)
  target_link_libraries(TestCudaFunction gtest pthread)
endif()
