#ifndef BIPARTITE_GRAPH_H
#define BIPARTITE_GRAPH_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stdlib.h"

typedef struct bipartite_graph {
  int *horizontal;
  int *vertical;
  int rows;
  int cols;
} bipartite_graph_t;

void bipartite_graph_clear_all(bipartite_graph_t *graph)
{
  for (int i = 0; i < graph->rows; i++) {
    graph->vertical[i] = -1;
  }
  for (int j = 0; j < graph->cols; j++) {
    graph->horizontal[j] = -1;
  }
}

void bipartite_graph_alloc(bipartite_graph_t *graph, int rows, int cols)
{
  graph->rows = rows;
  graph->cols = cols;
  graph->horizontal = (int *) calloc(cols, sizeof(int));
  graph->vertical = (int *) calloc(rows, sizeof(int));
}

void bipartite_graph_free(bipartite_graph_t *graph)
{
  free(graph->horizontal);
  free(graph->vertical);
}

void bipartite_graph_clear(bipartite_graph_t *graph, int i, int j)
{
  graph->horizontal[j] = -1;
  graph->vertical[i] = -1;
}

void bipartite_graph_connect(bipartite_graph_t *graph, int i, int j)
{
  graph->vertical[i] = j;
  graph->horizontal[j] = i;
}

#ifdef __cplusplus
}
#endif

#endif
