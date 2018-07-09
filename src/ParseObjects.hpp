#pragma once

#include "Munkres.hpp"
#include "ConnectParts.hpp"
#include "PafCostGraph.hpp"
#include "FindPeaks.hpp"
#include "ParserConfig.hpp"

#define PAF_I_DIM 1
#define PAF_J_DIM 0

std::vector<std::unordered_map<int, std::pair<int, int>>> parseObjects(float *cmap, float *paf, const ParserConfig &config)
{
  std::vector<std::unordered_map<int, std::pair<int, int>>> objects;
  std::vector<std::vector<std::pair<int, int>>> peaks;
  std::vector<int> part_counts;
  std::vector<PairGraph> pair_graphs;

  int height = config.map_height;
  int width = config.map_width;
  int size = config.map_width * config.map_height;

  // detect peaks
  for (int i = 0; i < config.num_parts; i++)
  {
    Matrix<float> cmap_i(cmap + i * size, height, width);
    peaks.push_back(findPeaks(cmap_i, config.peak_threshold));
    part_counts.push_back(peaks[peaks.size() - 1].size());
  }

  // compute part connections
  for (size_t i = 0; i < config.topology.size(); i++)
  {
    std::pair<Matrix<float>, Matrix<float>> paf_i = {
      Matrix<float>(paf + (2 * i + PAF_I_DIM) * size, height, width),
      Matrix<float>(paf + (2 * i + PAF_J_DIM) * size, height, width)
    };

    // compute cost graph
    auto cost_graph = pafCostGraph(
        paf_i, 
        { peaks[config.topology[i].first], peaks[config.topology[i].second] },
        config.paf_cost_num_samples
    );

    // find optimal part-part assignment via munkres algorithm
    auto pair_graph = PairGraph(cost_graph.nrows, cost_graph.ncols);
    munkres(cost_graph, pair_graph);
    pair_graphs.push_back(pair_graph);
  }

  // connect parts to form objects (contains reference indicies, not actual peak cooridnates)
  auto objects_ = ConnectParts(part_counts, pair_graphs, config.topology);  

  objects.resize(objects_.size());

  // parse object indicies and peaks coordinates
  for (size_t i = 0; i < objects.size(); i++)
  {
    for (auto &p : objects_[i])
    {
      objects[i].insert({p.first, peaks[p.first][p.second]});
    }
  }

  return objects;
}
