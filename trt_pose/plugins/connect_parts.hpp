#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include "munkres.hpp"
#include "utils/PairGraph.hpp"

std::unordered_map<int, int> searchPart(
    std::pair<int, int> node,
    const std::vector<PairGraph> &graphs,
    const std::vector<std::pair<int, int>> &topology,
    std::vector<std::vector<bool>> &visited);

// match 
std::vector<std::unordered_map<int, int>> connectParts(
    const std::vector<int> &part_counts,
    const std::vector<PairGraph> &graphs,
    const std::vector<std::pair<int, int>> &topology);

// returns NxOxC -1 if part absent, 1 if present
std::vector<torch::Tensor> connect_parts(torch::Tensor score_graph, torch::Tensor topology, torch::Tensor counts, float score_threshold, int max_object_count);