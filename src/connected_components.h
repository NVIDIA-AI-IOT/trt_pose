#ifndef CONNECTED_COMPONENTS_H
#define CONNECTED_COMPONENTS_H

#include "matrix.h"
#include "vector.h"
#include "component.h"

#ifdef __cplusplus
extern "C" {
#endif

int connected_components(imatrix_t *components, int *part_counts, imatrix_t *assignment_graphs,
    ivector2_t *assignment_graph_indices, int num_assignment_graphs);

#ifdef __cplusplus
}
#endif

#endif
