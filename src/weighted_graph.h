#ifndef WEIGHTED_GRAPH_H
#define WEIGHTED_GRAPH_H

#include "tensor.h"

typedef struct weighted_graph {
  float *data;
  int rows;
  int cols;
} weighted_graph_t;

inline float weighted_graph_at(weighted_graph_t *self, int row, int col)
{
  return self->data[IDX_2D(row, col, self->cols)];
}

#endif
