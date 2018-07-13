#include "gtest/gtest.h"

#include "../src/ConnectParts.hpp"

TEST(connect_parts, ShouldWorkA)
{
  std::vector<int> parts(2);
  parts[0] = 2;
  parts[1] = 2;

  std::vector<std::pair<int, int>> topology = {
    { 0, 1 },
  };

  PairGraph graph0(2, 2);
  graph0.set(0, 0);
  graph0.set(1, 1);

  auto components = connectParts(parts, { graph0 }, topology);
  ASSERT_EQ(2, components.size());
}

TEST(connect_parts, ShouldWorkB)
{
  std::vector<int> parts(3);
  parts[0] = 3;
  parts[1] = 3;
  parts[2] = 3;

  std::vector<std::pair<int, int>> topology = {
    { 0, 1 },
    { 1, 2}
  };

  PairGraph graph0(3, 3);
  graph0.set(0, 0);
  graph0.set(1, 1);
  graph0.set(2, 2);
  PairGraph graph1(3, 3);
  graph1.set(0, 0);
  graph1.set(1, 1);
  graph1.set(2, 2);

  auto components = connectParts(parts, { graph0, graph1 }, topology);
  ASSERT_EQ(3, components.size());
}

TEST(connect_parts, ShouldWorkC)
{
  std::vector<int> parts(3);
  parts[0] = 3;
  parts[1] = 3;
  parts[2] = 3;

  std::vector<std::pair<int, int>> topology = {
    { 0, 1 },
    { 1, 2 }
  };

  PairGraph graph0(3, 3);
  graph0.set(0, 1);
  graph0.set(1, 0);
  graph0.set(2, 2);
  PairGraph graph1(3, 3);
  graph1.set(0, 0);
  graph1.set(1, 1);
//graph1.set(2, 2);

  auto components = connectParts(parts, { graph0, graph1 }, topology);
  ASSERT_EQ(4, components.size());
}

#ifndef EXCLUDE_MAIN
int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif
