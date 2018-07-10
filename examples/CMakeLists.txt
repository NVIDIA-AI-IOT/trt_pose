find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

add_executable(trt_pose_example trt_pose_example.cpp)
target_link_libraries(trt_pose_example trt_pose argus_camera ${OpenCV_LIBS} ${CUDA_LIBRARIES})
