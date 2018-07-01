cmake_minimum_required(VERSION 3.5)
project(trt_pose)

set(CMAKE_CXX_FLAGS -std=c++11)
set(CUDA_NVCC_FLASGS -std=c++11)

find_package(CUDA REQUIRED)

cuda_add_library(trt_pose SHARED
)

if (BUILD_TESTS)
  find_package(GTest REQUIRED)

  #cuda_add_executable(test_tensor_transpose test/test_tensor_transpose.cu)
  #target_link_libraries(test_tensor_transpose gtest pthread trt_pose ${CUDA_CUBLAS_LIBRARIES})

endif()
