#pragma once

namespace trt_pose {
namespace parse {

void paf_score_graph_out_hw(float *score_graph, // MxM
                            const float *paf_i, // HxW
                            const float *paf_j, // HxW
                            const int counts_a, const int counts_b,
                            const float *peaks_a, // Mx2
                            const float *peaks_b, // Mx2
                            const int H, const int W, const int M,
                            const int num_integral_samples);

void paf_score_graph_out_khw(float *score_graph,  // KxMxM
                             const int *topology, // Kx4
                             const float *paf,    // 2KxHxW
                             const int *counts,   // C
                             const float *peaks,  // CxMx2
                             const int K, const int C, const int H, const int W,
                             const int M, const int num_integral_samples);

void paf_score_graph_out_nkhw(float *score_graph,  // NxKxMxM
                              const int *topology, // Kx4
                              const float *paf,    // Nx2KxHxW
                              const int *counts,   // NxC
                              const float *peaks,  // NxCxMx2
                              const int N, const int K, const int C,
                              const int H, const int W, const int M,
                              const int num_integral_samples);

} // namespace parse
} // namespace trt_pose
