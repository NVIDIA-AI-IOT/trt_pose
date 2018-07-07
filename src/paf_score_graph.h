#ifndef BIPARTITE_GRAPH_H
#define BIPARTITE_GRAPH_H

#define PAF_I_IDX 0
#define PAF_J_IDX 1

#ifdef __cplusplus
extern "C" {
#endif

void paf_score_graph(
    int *peak_counts, int **peak_ptrs, int peak_max_count,
    float *paf, int paf_channels, int paf_height, int paf_width,
    int *paf_cmap_pairs,
    float *paf_score_graph, int num_samples);

#ifdef __cplusplus
}
#endif

#endif // BIPARTITE_GRAPH_H
