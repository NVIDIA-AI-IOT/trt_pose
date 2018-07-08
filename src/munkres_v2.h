#ifndef MUNKRES_V2_H
#define MUNKRES_V2_H

#include "matrix.h"
#include "stdlib.h"
#include "bipartite_graph.h"

void munkres_v2_sub_min_row(matrix_t *cost_graph)
{
  for (int i = 0; i < cost_graph->rows; i++) 
  { 
    float min = matrix_at(cost_graph, i, 0);
    for (int j = 0; j < cost_graph->cols; j++) 
    {
      if (matrix_at(cost_graph, i, j) < min) 
      {
        min = matrix_at(cost_graph, i, j);
      }
    }
    for (int j = 0; j < cost_graph->cols; j++)
    {
      *matrix_at_mutable(cost_graph, i, j) -= min;
    }
  }
}

void munkres_v2_sub_min_col(matrix_t *cost_graph)
{
  for (int j = 0; j < cost_graph->cols; j++) 
  { 
    float min = matrix_at(cost_graph, 0, j);
    for (int i = 0; i < cost_graph->rows; i++) 
    {
      if (matrix_at(cost_graph, i, j) < min) 
      {
        min = matrix_at(cost_graph, i, j);
      }
    }
    for (int i = 0; i < cost_graph->rows; i++)
    {
      *matrix_at_mutable(cost_graph, i, j) -= min;
    }
  }
}

void munkres_v2_step_1(matrix_t *cost_graph, bipartite_graph_t *star_graph)
{
  for (int i = 0; i < cost_graph->rows; i++) 
  {
    for (int j = 0; j < cost_graph->cols; j++) 
    {
      if (matrix_at(cost_graph, i, j) == 0 && star_graph->vertical[i] < 0 && star_graph->horizontal[j] < 0)
      {
        bipartite_graph_connect(star_graph, i, j); 
      }
    }
  }
}

// 1 if exit, 0 otherwise
int munkres_v2_step_2(bipartite_graph_t *star_graph, int *c1)
{
  int count = 0;
  int k = star_graph->rows < star_graph->cols ? star_graph->rows : star_graph->cols;
  for (int j = 0; j < star_graph->cols; j++) 
  {
    if (star_graph->horizontal[j] >= 0) 
    {
      c1[j] = 1;
      count++;
    }
  }
  return count >= k;
}

int munkres_v2_step_3(matrix_t *cost_graph, bipartite_graph_t *star_graph, bipartite_graph_t *prime_graph, int *c0, int *c1, int *p0, int *p1)
{
  for (int i = 0; i < star_graph->rows; i++) 
  {
    for (int j = 0; j < star_graph->cols; j++)
    {
      if (!c0[i] && !c1[j] && matrix_at(cost_graph, i, j) == 0)
      {
        bipartite_graph_connect(prime_graph, i, j);
        if (star_graph->vertical[i] < 0) 
        {
          *p0 = i;
          *p1 = j;
          return 1; // goto step 4
        }
        else
        {
          c0[i] = 1;
          c1[j] = 0;
        }
      }
    }
  }
  return false;
}

void munkres_v2_step_4(bipartite_graph_t *star_graph, bipartite_graph_t *prime_graph, int *c0, int *c1, int p0, int p1)
{
  while (star_graph->horizontal[p1] >= 0)
  {
    // cache star row
    int s0 = star_graph->horizontal[p1];

    // clear star and star prime
    bipartite_graph_clear(star_graph, s0, p1);
    bipartite_graph_connect(star_graph, p0, p1);

    // find new prime
    p0 = s0;
    p1 = prime_graph->vertical[s0]; // prime in s0's row
  }
  for (int i = 0; i < star_graph->rows; i++) 
  {
    c0[i] = 0;
  }
  for (int j = 0; j < star_graph->cols; j++)
  {
    c1[j] = 0;
  }
  bipartite_graph_clear_all(prime_graph);
}

void munkres_v2_step_5(matrix_t *cost_graph, int *c0, int *c1)
{
  int valid = 0;
  float min;
  for (int i = 0; i < cost_graph->rows; i++)
  {
    for (int j = 0; j < cost_graph->cols; j++)
    {
      if (!c0[i] && !c1[j]) 
      {
        float val = matrix_at(cost_graph, i, j);
        if (!valid) 
        {
          min = val;
          valid = 1;
        } 
        else if (val < min)
        {
          min = val;
        }
      }
    }
  }

  // sub min
  for (int i = 0; i < cost_graph->rows; i++)
  {
    for (int j = 0; j < cost_graph->cols; j++)
    {
      if (c0[i]) 
      {
        *matrix_at_mutable(cost_graph, i, j) += min;
      } 
      else if (!c1[j]) 
      {
        *matrix_at_mutable(cost_graph, i, j) -= min;
      }
    }
  }
}

void munkres_v2(matrix_t *cost_graph, bipartite_graph_t *star_graph) 
{
  int *c0 = (int*)calloc(cost_graph->rows, sizeof(int));
  int *c1 = (int*)calloc(cost_graph->cols, sizeof(int));
  bipartite_graph_clear_all(star_graph);
  bipartite_graph_t prime_graph;
  bipartite_graph_alloc(&prime_graph, cost_graph->rows, cost_graph->cols);
  bipartite_graph_clear_all(&prime_graph);

  int step = 1;
  int done = 0;

  // preliminaries
  if (cost_graph->rows > cost_graph->cols) 
  {
    munkres_v2_sub_min_col(cost_graph);
  } 
  else if (cost_graph->rows == cost_graph->cols) {
    munkres_v2_sub_min_row(cost_graph);
    munkres_v2_sub_min_col(cost_graph);
  } 
  else {
    munkres_v2_sub_min_row(cost_graph);
  }

  int p0, p1;
 
  while (!done) 
  {
    switch(step)
    {
      case 1:
        munkres_v2_step_1(cost_graph, star_graph);
      case 2:
        if (munkres_v2_step_2(star_graph, c1))
        {
          done = 1;
          break;
        }
      case 3:
        if (!munkres_v2_step_3(cost_graph, star_graph, &prime_graph, c0, c1, &p0, &p1)) 
        {
          step = 5;
          break;
        }
      case 4:
        munkres_v2_step_4(star_graph, &prime_graph, c0, c1, p0, p1);
        step = 2;
        break;
      case 5:
        munkres_v2_step_5(cost_graph, c0, c1);
        break;
    }
  }

  free(c0);
  free(c1);
  bipartite_graph_free(&prime_graph);
}

#endif
