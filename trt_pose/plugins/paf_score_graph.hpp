#pragma once

void paf_score_graph_out_hw(float *score_graph, // MxM
                        const float *paf_i, // HxW
                        const float *paf_j, // HxW
                        const int counts_a, const int counts_b,
                        const float *peaks_a, // Mx2
                        const float *peaks_b, // Mx2
                        const int H, const int W, const int M,
                        const int num_integral_samples);
