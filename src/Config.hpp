#pragma once

#include <vector>

class Config
{
public:
  std::vector<std::pair<int, int>> topology;// size should match paf channels
  int num_parts; // number of cmap channels
  int map_height;
  int map_width;
  float peak_threshold;
  int paf_cost_num_samples;
};
