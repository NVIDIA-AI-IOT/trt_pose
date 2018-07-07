#ifndef PAF_COST_GRAPH
#define PAF_COST_GRAPH

#include "matrix.h"
#include "peak.h"
#include "vector.h"

#ifdef __cplusplus
extern "C" {
#endif

void paf_cost_graph(matrix_t *cost_graph,
    matrix_t *paf_i, matrix_t *paf_j, vector2_t *src_peaks, 
    int num_src_peaks, vector2_t *dst_peaks, int num_dst_peaks, int num_samples);

#ifdef __cplusplus
}
#endif

#endif // PAF_COST_GRAPH
