cmake_minimum_required(VERSION 3.6)
project(trt_pose)

add_library(trt_pose SHARED
  trt_pose/plugins/find_peaks.cpp
  trt_pose/plugins/refine_peaks.cpp
  trt_pose/plugins/paf_score_graph.cpp
  trt_pose/plugins/munkres.cpp
  trt_pose/plugins/connect_parts.cpp
)

add_executable(trt_pose_test_all
  trt_pose/plugins/test_all.cpp
)
target_link_libraries(trt_pose_test_all trt_pose)
