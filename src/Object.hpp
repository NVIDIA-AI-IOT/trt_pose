#pragma once

#include <vector>
#include <unordered_map>

#include "Gaussian.hpp"

class Object
{
public:
  std::unordered_map<int, std::pair<int, int>> peaks;
  std::unordered_map<int, Gaussian> gaussians;
};
