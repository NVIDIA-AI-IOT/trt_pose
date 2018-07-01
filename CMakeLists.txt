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
  src/tensor_transpose.h
)

if (BUILD_TESTS)
  find_package(GTest REQUIRED)

  add_executable(test_tensor test/test_tensor.cpp)
  target_link_libraries(test_tensor gtest pthread trt_pose)

  cuda_add_executable(test_tensor_transpose test/test_tensor_transpose.cu)
  target_link_libraries(test_tensor_transpose gtest pthread trt_pose ${CUDA_CUBLAS_LIBRARIES})

  cuda_add_executable(test_tensor_matmul test/test_tensor_matmul.cu)
  target_link_libraries(test_tensor_matmul gtest pthread trt_pose ${CUDA_CUBLAS_LIBRARIES})
endif()
