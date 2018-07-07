#ifndef GRAPH_H
#define GRAPH_H

#include "tensor.h"

typedef struct graph {
  int *data;
  int rows;
  int cols;
} graph_t;

inline int graph_at(graph_t *self, int row, int col)
{
  return self->data[IDX_2D(row, col, self->cols)];
}

#endif
