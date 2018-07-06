cmake_minimum_required(VERSION 3.5)
project(trt_pose)

set(CMAKE_CXX_FLAGS -std=c++11)

set(CUDA_NVCC_FLASGS -std=c++11)

#if (CMAKE_BUILD_TYPE == DEBUG)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -G -g)
  #endif()

find_package(CUDA REQUIRED)

cuda_add_library(trt_pose SHARED
)

target_link_libraries(trt_pose ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARY})

if (BUILD_TESTS)
  find_package(GTest REQUIRED)

  cuda_add_executable(test_all test/test_all.cu)
  target_link_libraries(test_all gtest pthread trt_pose ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARY})

  cuda_add_executable(test_hungarian_assign test/test_hungarian_assign.cu)
  target_link_libraries(test_hungarian_assign gtest pthread trt_pose ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARY})

endif()
