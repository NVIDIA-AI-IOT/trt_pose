cmake_minimum_required(VERSION 3.5)
project(trt_pose)

set(CMAKE_CXX_FLAGS -std=c++11)
set(CUDA_NVCC_FLASGS -std=c++11)

find_package(CUDA REQUIRED)

cuda_add_library(trt_pose SHARED
  src/matrix_fill.cu
)

if (BUILD_TESTS)
  find_package(GTest REQUIRED)

  cuda_add_executable(test_matrix test/test_matrix.cu)
  target_link_libraries(test_matrix gtest pthread trt_pose ${CUDA_CUBLAS_LIBRARIES})

  cuda_add_executable(test_matrix_multiply test/test_matrix_multiply.cu)
  target_link_libraries(test_matrix_multiply gtest pthread trt_pose ${CUDA_CUBLAS_LIBRARIES})

  cuda_add_executable(test_matrix_fill test/test_matrix_fill.cu)
  target_link_libraries(test_matrix_fill gtest pthread trt_pose ${CUDA_CUBLAS_LIBRARIES})
endif()
