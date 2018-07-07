#ifndef GRAPH_H
#define GRAPH_H

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct graph {
  int *data;
  int rows;
  int cols;
} graph_t;

extern inline int graph_at(graph_t *self, int row, int col)
{
  return self->data[IDX_2D(row, col, self->cols)];
}

#ifdef __cplusplus
}
#endif

#endif
