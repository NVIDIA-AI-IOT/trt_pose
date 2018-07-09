#pragma once

#include <string>
#include <vector>
#include <unordered_map>

class Config
{
public:
  int partIndex(std::string name) {
    for (size_t i = 0; i < part_names.size(); i++) 
    {
      if (part_names[i] == name)
      {
        return i;
      }
    }
    return -1;
  }

  std::string partName(int idx) {
    return part_names[idx];
  }

  std::vector<std::string> part_names;
  std::vector<std::pair<int, int>> topology;// size should match paf channels
  std::string trt_input_name;
  std::string trt_paf_name;
  std::string trt_cmap_name;
  float peak_threshold;
  int paf_cost_num_samples;
};
