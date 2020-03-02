#include "paf_score_graph.hpp"
#include <cmath>

#define EPS 1e-5

namespace trt_pose {
namespace parse {

void paf_score_graph_out_hw(float *score_graph, // MxM
                            const float *paf_i, // HxW
                            const float *paf_j, // HxW
                            const int counts_a, const int counts_b,
                            const float *peaks_a, // Mx2
                            const float *peaks_b, // Mx2
                            const int H, const int W, const int M,
                            const int num_integral_samples) {
  for (int a = 0; a < counts_a; a++) {
    // compute point A
    float pa_i = peaks_a[a * 2] * H;
    float pa_j = peaks_a[a * 2 + 1] * W;

    for (int b = 0; b < counts_b; b++) {
      // compute point B
      float pb_i = peaks_b[b * 2] * H;
      float pb_j = peaks_b[b * 2 + 1] * W;

      // compute vector A->B
      float pab_i = pb_i - pa_i;
      float pab_j = pb_j - pa_j;

      // compute normalized vector A->B
      float pab_norm = sqrtf(pab_i * pab_i + pab_j * pab_j) + EPS;
      float uab_i = pab_i / pab_norm;
      float uab_j = pab_j / pab_norm;

      float integral = 0.;
      float increment = 1.f / num_integral_samples;

      for (int t = 0; t < num_integral_samples; t++) {
        // compute integral point T
        float progress = (float)t / ((float)num_integral_samples - 1);
        float pt_i = pa_i + progress * pab_i;
        float pt_j = pa_j + progress * pab_j;

        // convert to int
        // note: we do not need to subtract 0.5 when indexing, because
        // round(x - 0.5) = int(x)
        int pt_i_int = (int)pt_i;
        int pt_j_int = (int)pt_j;

        // skip point if out of bounds (will weaken integral)
        if (pt_i_int < 0)
          continue;
        if (pt_i_int >= H)
          continue;
        if (pt_j_int < 0)
          continue;
        if (pt_j_int >= W)
          continue;

        // get vector at integral point from PAF
        float pt_paf_i = paf_i[pt_i_int * W + pt_j_int];
        float pt_paf_j = paf_j[pt_i_int * W + pt_j_int];

        // compute dot product of normalized A->B with PAF vector at integral
        // point
        float dot = pt_paf_i * uab_i + pt_paf_j * uab_j;
        integral += dot;
      }

      integral /= num_integral_samples;
      score_graph[a * M + b] = integral;
    }
  }
}

void paf_score_graph_out_khw(float *score_graph,  // KxMxM
                             const int *topology, // Kx4
                             const float *paf,    // 2KxHxW
                             const int *counts,   // C
                             const float *peaks,  // CxMx2
                             const int K, const int C, const int H, const int W,
                             const int M, const int num_integral_samples) {
  for (int k = 0; k < K; k++) {
    float *score_graph_k = &score_graph[k * M * M];
    const int *tk = &topology[k * 4];
    const int paf_i_idx = tk[0];
    const int paf_j_idx = tk[1];
    const int cmap_a_idx = tk[2];
    const int cmap_b_idx = tk[3];
    const float *paf_i = &paf[paf_i_idx * H * W];
    const float *paf_j = &paf[paf_j_idx * H * W];

    const int counts_a = counts[cmap_a_idx];
    const int counts_b = counts[cmap_b_idx];
    const float *peaks_a = &peaks[cmap_a_idx * M * 2];
    const float *peaks_b = &peaks[cmap_b_idx * M * 2];

    paf_score_graph_out_hw(score_graph_k, paf_i, paf_j, counts_a, counts_b,
                           peaks_a, peaks_b, H, W, M, num_integral_samples);
  }
}

void paf_score_graph_out_nkhw(float *score_graph,  // NxKxMxM
                              const int *topology, // Kx4
                              const float *paf,    // Nx2KxHxW
                              const int *counts,   // NxC
                              const float *peaks,  // NxCxMx2
                              const int N, const int K, const int C,
                              const int H, const int W, const int M,
                              const int num_integral_samples) {
  for (int n = 0; n < N; n++) {
    paf_score_graph_out_khw(&score_graph[n * K * M * M], topology,
                            &paf[n * 2 * K * H * W], &counts[n * C],
                            &peaks[n * C * M * 2], K, C, H, W, M,
                            num_integral_samples);
  }
}

} // namespace parse
} // namespace trt_pose
