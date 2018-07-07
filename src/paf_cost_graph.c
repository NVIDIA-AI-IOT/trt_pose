#include "paf_cost_graph.h"

#include "math.h"
#include "tensor.h"

void paf_cost_graph(matrix_t *cost_graph,
    matrix_t *paf_i, matrix_t *paf_j, vector2_t *src_peaks, 
    int num_src_peaks, vector2_t *dst_peaks, int num_dst_peaks, int num_samples)
{
  for (int i = 0; i < num_src_peaks; i++) {

    vector2_t p0 = src_peaks[i];

    for (int j = 0; j < num_dst_peaks; j++) {

      vector2_t p1 = dst_peaks[j];

      vector2_t p01 = vector2_sub(p1, p0);
      vector2_t p01_normed = vector2_normalized(p01);

      for (int k = 0; k < num_samples; k++) {

        vector2_t sample_idx = vector2_add(p0, vector2_scale(p01, k / num_samples));

        vector2_t sample_val;
        sample_val.i = matrix_at(paf_i, sample_idx.i, sample_idx.j);
        sample_val.j = matrix_at(paf_j, sample_idx.i, sample_idx.j);

        // negative to convert from reward to cost matrix
        *matrix_at_mutable(cost_graph, i, j) = - vector2_dot(p01_normed, sample_val);
      }
    }
  }
}
