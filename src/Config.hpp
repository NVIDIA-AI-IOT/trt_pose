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

Config DEFAULT_HUMAN_POSE_CONFIG()
{
  Config config;
  config.trt_cmap_name = "Mconv7_stage6_L2";
  config.trt_paf_name = "Mconv7_stage6_L1";
  config.trt_input_name = "image";
  config.peak_threshold = 0.3;
  config.paf_cost_num_samples = 10;
  config.part_names = {
    "NOSE",
    "NECK",
    "RSHOULDER",
    "RELBOW",
    "RWRIST",
    "LSHOULDER",
    "LELBOW",
    "LWRIST",
    "RHIP",
    "RKNEE",
    "RANKLE",
    "LHIP",
    "LKNEE",
    "LANKLE",
    "REYE",
    "LEYE",
    "REAR",
    "LEAR",
  };
  config.topology = {
    { config.partIndex("NECK"), config.partIndex("RHIP") },
    { config.partIndex("RHIP"), config.partIndex("RKNEE") },
    { config.partIndex("RKNEE"), config.partIndex("RANKLE") },
    { config.partIndex("NECK"), config.partIndex("LHIP") },
    { config.partIndex("LHIP"), config.partIndex("LKNEE") },
    { config.partIndex("LKNEE"), config.partIndex("LANKLE") },
    { config.partIndex("NECK"), config.partIndex("RSHOULDER") },
    { config.partIndex("RSHOULDER"), config.partIndex("RELBOW") },
    { config.partIndex("RELBOW"), config.partIndex("RWRIST") },
    { config.partIndex("RSHOULDER"), config.partIndex("REAR") },
    { config.partIndex("NECK"), config.partIndex("LSHOULDER") },
    { config.partIndex("LSHOULDER"), config.partIndex("LELBOW") },
    { config.partIndex("LELBOW"), config.partIndex("LWRIST") },
    { config.partIndex("LSHOULDER"), config.partIndex("LEAR") },
    { config.partIndex("NECK"), config.partIndex("NOSE") },
    { config.partIndex("NOSE"), config.partIndex("REYE") },
    { config.partIndex("NOSE"), config.partIndex("LEYE") },
    { config.partIndex("REYE"), config.partIndex("REAR") },
    { config.partIndex("LEYE"), config.partIndex("LEAR") },
  };
  return config;
}
