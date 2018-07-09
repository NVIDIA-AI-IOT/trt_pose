#include "gtest/gtest.h"

#include "../src/Matrix.hpp"
#include "../src/PafCostGraph.hpp"

TEST(paf_cost_graph, valid)
{
  const int n = 4;
  const int m = 4;

  float data0[n * m] = {
    0, 1, 0, 0,
    0, 1, 0, 0,
    0, 1, 0, 0,
    0, 0, 0, 0
  };

  float data1[n * m] = {
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 1, 1, 1
  };

  Matrix<float> paf0(data0, n, m);
  Matrix<float> paf1(data1, n, m);

  auto cost_graph = pafCostGraph({paf0, paf1}, { {{0,1}, {3,1}}, {{2,1}, {3,3}} }, 5);
  ASSERT_NEAR(-1.0f, cost_graph.at(0, 0), 0.01);
  ASSERT_NEAR(-1.0f, cost_graph.at(1, 1), 0.01);
}

#ifndef EXCLUDE_MAIN
int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif
