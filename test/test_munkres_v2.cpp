#include "gtest/gtest.h"

#include "test_utils.h"

#include "../src/munkres_v2.h"
#include "../src/tensor.h"
#include "../src/bipartite_graph.h"

float solution_sum(matrix_t *cost_graph, bipartite_graph_t *star_graph)
{
  float sum = 0;
  for (int i = 0; i < cost_graph->rows; i++)
  {
    sum += matrix_at(cost_graph, i, star_graph->vertical[i]);
  }
  return sum;
}

TEST(test_munkres_v2, Correct0)
{
  const int n = 3;
  const int m = 4;
  float cost_graph_data[n * m] = {
    1, 2, 3, 2,
    2, 1, 2, 3,
    2, 3, 1, 4
  };
  float cost_graph_data_old[n * m] = {
    1, 2, 3, 2,
    2, 1, 2, 3,
    2, 3, 1, 4
  };
  matrix_t cost_graph = { cost_graph_data, n, m };
  bipartite_graph_t star_graph;
  bipartite_graph_alloc(&star_graph, n, m);
  bipartite_graph_clear_all(&star_graph);

  munkres_v2(&cost_graph, &star_graph);
  cost_graph.data = cost_graph_data_old;
  
  ASSERT_EQ(3, solution_sum(&cost_graph, &star_graph));

  bipartite_graph_free(&star_graph);
}

TEST(test_munkres_v2, Correct1)
{
  const int n = 3;
  const int m = 4;
  float cost_graph_data[n * m] = {
    1, 2, 3, 2,
    1, 2, 2, 3,
    1, 3, 2, 4
  };
  float cost_graph_data_old[n * m] = {
    1, 2, 3, 2,
    1, 2, 2, 3,
    1, 3, 2, 4
  };
  matrix_t cost_graph = { cost_graph_data, n, m };
  bipartite_graph_t star_graph;
  bipartite_graph_alloc(&star_graph, n, m);
  bipartite_graph_clear_all(&star_graph);

  munkres_v2(&cost_graph, &star_graph);
  cost_graph.data = cost_graph_data_old;

  ASSERT_EQ(5, solution_sum(&cost_graph, &star_graph));
  
  bipartite_graph_free(&star_graph);
}

TEST(test_munkres_v2, Correct2)
{
  const int n = 3;
  const int m = 4;
  float cost_graph_data[n * m] = {
    3, 2, 3, 2,
    3, 3, 3, 3,
    2, 3, 2, 4
  };
  float cost_graph_data_old[n * m] = {
    3, 2, 3, 2,
    3, 3, 3, 3,
    2, 3, 2, 4
  };
  matrix_t cost_graph = { cost_graph_data, n, m };
  bipartite_graph_t star_graph;
  bipartite_graph_alloc(&star_graph, n, m);
  bipartite_graph_clear_all(&star_graph);

  munkres_v2(&cost_graph, &star_graph);
  cost_graph.data = cost_graph_data_old;
  
  ASSERT_EQ(7, solution_sum(&cost_graph, &star_graph));
  
  bipartite_graph_free(&star_graph);
}

#ifndef EXCLUDE_MAIN
int main(int argc, char *argv[]) 
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif
