cmake_minimum_required(VERSION 3.5)
project(trt_pose)

set(CMAKE_CXX_FLAGS -std=c++11)
set(CUDA_NVCC_FLASGS -std=c++11)

find_package(CUDA REQUIRED)

cuda_add_library(trt_pose SHARED
  src/tensor.h
  src/tensor.cpp
  src/tensor.cu
  src/tensor_matmul.h
)

if (BUILD_TESTS)
  find_package(GTest REQUIRED)

  add_executable(test_tensor test/test_tensor.cpp)
  target_link_libraries(test_tensor gtest pthread trt_pose)
endif()
