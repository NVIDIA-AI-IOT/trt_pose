cmake_minimum_required(VERSION 3.6)
project(trt_pose)

add_library(trt_pose SHARED
  trt_pose/parse/find_peaks.cpp
  trt_pose/parse/refine_peaks.cpp
  trt_pose/parse/paf_score_graph.cpp
  trt_pose/parse/munkres.cpp
  trt_pose/parse/connect_parts.cpp
)

add_executable(trt_pose_test_all
  trt_pose/parse/test_all.cpp
)
target_link_libraries(trt_pose_test_all trt_pose)
