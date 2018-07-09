#pragma once

#include "Munkres.hpp"
#include "ConnectParts.hpp"
#include "PafCostGraph.hpp"
#include "FindPeaks.hpp"
#include "ParserConfig.hpp"

#define PAF_I_DIM 1
#define PAF_J_DIM 0

std::vector<std::unordered_map<int, std::pair<int, int>>> parseObjects(float *cmap, float *paf, const ParserConfig &config);
