#pragma once

#include <cstring>

namespace trt_pose {
namespace parse {

std::size_t assignment_out_workspace(const int M);

void assignment_out(int *connections,         // 2xM
                    const float *score_graph, // MxM
                    const int count_a, const int count_b, const int M,
                    const float score_threshold, void *workspace);

void assignment_out_k(int *connections,         // Kx2xM
                       const float *score_graph, // KxMxM
                       const int *topology,      // Kx4
                       const int *counts,        // C
                       const int K, const int M, const float score_threshold,
                       void *workspace);

void assignment_out_nk(int *connections,         // NxKx2xM
                        const float *score_graph, // NxKxMxM
                        const int *topology,      // Kx4
                        const int *counts,        // NxC
                        const int N, const int C, const int K, const int M,
                        const float score_threshold, void *workspace);

} // namespace parse
} // namespace trt_pose
