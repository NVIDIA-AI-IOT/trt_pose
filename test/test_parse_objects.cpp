#include "gtest/gtest.h"

#include "../src/Matrix.hpp"
#include "../src/PafCostGraph.hpp"
#include "../src/ParseObjects.hpp"
#include "../src/ParserConfig.hpp"

TEST(parse_objects, valid)
{
  ParserConfig config;
  config.topology = {
    { 0, 1 }
  };
  config.num_parts = 2;
  config.map_height = 4;
  config.map_width = 4;
  config.peak_threshold = 0.5;
  config.paf_cost_num_samples = 5;


  float paf[] = {
    
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 1, 1, 1,

    0, 1, 0, 0,
    0, 1, 0, 0,
    0, 1, 0, 0,
    0, 0, 0, 0,

  };


  float cmap[] = {
    0, 1, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 1, 0, 0,

    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 0, 1
  };

  auto objects = parseObjects(cmap, paf, config);
  ASSERT_EQ(2, objects.size());
}

#ifndef EXCLUDE_MAIN
int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif
