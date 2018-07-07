#include "paf_score_graph.h"

#include "math.h"
#include "tensor.h"

void paf_score_graph(
    int *peak_counts, int **peak_ptrs, int peak_max_count,
    float *paf, int paf_channels, int paf_height, int paf_width,
    int *paf_cmap_pairs,
    float *paf_score_graph, int num_samples)
{
  int paf_count = paf_channels / 2;
  for (int k = 0; k < paf_count; k++) {
    int c0 = paf_cmap_pairs[2 * k];
    int c1 = paf_cmap_pairs[2 * k + 1];
    float *paf_score_graph_k = paf_score_graph + peak_max_count * peak_max_count;
    for (int i = 0; i < peak_counts[c0]; i++) {
      float p0_i = UNRAVEL_2D_i(peak_ptrs[c0][i], paf_width);
      float p0_j = UNRAVEL_2D_j(peak_ptrs[c0][i], paf_width);

      for (int j = 0; j < peak_counts[c1]; j++) {
        float p1_i = UNRAVEL_2D_i(peak_ptrs[c1][j], paf_width);
        float p1_j = UNRAVEL_2D_j(peak_ptrs[c1][j], paf_width);

        // compute normalized vector
        float p01_i = p1_i - p0_i;
        float p01_j = p1_j - p0_j;
        float p01_mag = sqrtf(p01_i * p01_i + p01_j * p01_j);
        float p01_i_normed = p01_i / p01_mag;
        float p01_j_normed = p01_j / p01_mag;

        for (int n = 0; n < num_samples; n++) {
          float pS_i = p0_i + ((float) n) * p01_i / ((float) num_samples); // sample i coordinate
          float pS_j = p0_j + ((float) n) * p01_j / ((float) num_samples); // sample j coordiante
          float pS_val_i = paf[IDX_3D(2 * k + PAF_I_IDX, (int) pS_i, (int) pS_j, paf_height, paf_width)]; // paf i value at sample i coordinate
          float pS_val_j = paf[IDX_3D(2 * k + PAF_J_IDX, (int) pS_i, (int) pS_j, paf_height, paf_width)]; // paf j value at sample j coordinate
          float dot_product = pS_val_i * p01_i_normed + pS_val_j * p01_j_normed;
          paf_score_graph_k[IDX_2D(i, j, peak_counts[c1])] = dot_product;
        }
      }
    }
  }
}
