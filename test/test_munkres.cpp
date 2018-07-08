#include "gtest/gtest.h"

#include "../src/Matrix.hpp"
#include "../src/PairGraph.hpp"
#include "../src/Munkres.hpp"

TEST(munkres, Test3x4_A)
{
  const int n = 3;
  const int m = 4;

  float cost_graph_data[n * m] = {
    1, 2, 3, 4,
    2, 1, 2, 3,
    2, 3, 1, 4
  };

  Matrix<float> cost_graph(cost_graph_data, n, m);
  Matrix<float> cost_graph_old(n, m);
  cost_graph_old.copy(cost_graph);
  PairGraph star_graph(n, m);

  munkres(cost_graph, star_graph);

  ASSERT_EQ(0, star_graph.colForRow(0));
  ASSERT_EQ(1, star_graph.colForRow(1));
  ASSERT_EQ(2, star_graph.colForRow(2));
  ASSERT_EQ(0, star_graph.rowForCol(0));
  ASSERT_EQ(1, star_graph.rowForCol(1));
  ASSERT_EQ(2, star_graph.rowForCol(2));
  ASSERT_EQ(true, star_graph.isPair(0, 0));
  ASSERT_EQ(true, star_graph.isPair(1, 1));
  ASSERT_EQ(true, star_graph.isPair(2, 2));
  ASSERT_EQ(3, cost_graph_old.sumIndices(star_graph.pairs()));
}

TEST(munkres, Test3x4_B)
{
  const int n = 3;
  const int m = 4;

  float cost_graph_data[n * m] = {
    1, 2, 3, 4,
    1, 2, 3, 3,
    1, 3, 3, 4
  };

  Matrix<float> cost_graph(cost_graph_data, n, m);
  Matrix<float> cost_graph_old(n, m);
  cost_graph_old.copy(cost_graph);
  PairGraph star_graph(n, m);

  munkres(cost_graph, star_graph);

  ASSERT_EQ(6, cost_graph_old.sumIndices(star_graph.pairs()));
}

TEST(munkres, Test4x3_A)
{
  const int n = 4;
  const int m = 3;

  float cost_graph_data[n * m] = {
    1, 2, 9, 
    1, 2, 7,
    4, 3, 3,
    1, 8, 4
  };

  Matrix<float> cost_graph(cost_graph_data, n, m);
  Matrix<float> cost_graph_old(n, m);
  cost_graph_old.copy(cost_graph);
  PairGraph star_graph(n, m);

  munkres(cost_graph, star_graph);

  ASSERT_EQ(6, cost_graph_old.sumIndices(star_graph.pairs()));
}

#ifndef EXCLUDE_MAIN
int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif
