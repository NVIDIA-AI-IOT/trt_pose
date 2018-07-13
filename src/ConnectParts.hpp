#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include "PairGraph.hpp"

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

