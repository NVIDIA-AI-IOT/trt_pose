cmake_minimum_required(VERSION 3.5)
project(trt_pose)

set(CMAKE_CXX_FLAGS -std=c++11)

set(CUDA_NVCC_FLASGS -std=c++11)

#if (CMAKE_BUILD_TYPE == DEBUG)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -G -g)
  #endif()

find_package(CUDA REQUIRED)

cuda_add_library(trt_pose SHARED
  src/matrix_fill.cu
  src/matrix_solve.cu
  src/matrix_peak_threshold.cu
  src/gaussian_fit.cu
)

target_link_libraries(trt_pose ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARY})

if (BUILD_TESTS)
  find_package(GTest REQUIRED)

  cuda_add_executable(test_all test/test_all.cu)
  target_link_libraries(test_all gtest pthread trt_pose ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARY})

  cuda_add_executable(test_matrix test/test_matrix.cu)
  target_link_libraries(test_matrix gtest pthread trt_pose ${CUDA_CUBLAS_LIBRARIES})

  cuda_add_executable(test_matrix_multiply test/test_matrix_multiply.cu)
  target_link_libraries(test_matrix_multiply gtest pthread trt_pose ${CUDA_CUBLAS_LIBRARIES})

  cuda_add_executable(test_matrix_fill test/test_matrix_fill.cu)
  target_link_libraries(test_matrix_fill gtest pthread trt_pose ${CUDA_CUBLAS_LIBRARIES})

  cuda_add_executable(test_matrix_solve test/test_matrix_solve.cu)
  target_link_libraries(test_matrix_solve gtest pthread trt_pose ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARY})

  cuda_add_executable(test_matrix_peak_threshold test/test_matrix_peak_threshold.cu)
  target_link_libraries(test_matrix_peak_threshold gtest pthread trt_pose ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARY})

  cuda_add_executable(test_gaussian_fit test/test_gaussian_fit.cu)
  target_link_libraries(test_gaussian_fit gtest pthread trt_pose ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARY})
  
  cuda_add_executable(test_matrix_add test/test_matrix_add.cu)
  target_link_libraries(test_matrix_add gtest pthread trt_pose ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARY})
endif()
