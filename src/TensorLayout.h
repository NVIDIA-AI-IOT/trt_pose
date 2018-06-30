#pragma once

#include <vector>
#include <cstdint>

std::vector<int64_t> DefaultStrides(std::vector<uint64_t> sizes);

struct TensorLayout
{
  std::vector<uint64_t> sizes;
  std::vector<int64_t> strides;

  TensorLayout(std::vector<uint64_t> sizes);
  TensorLayout(std::vector<uint64_t> sizes, std::vector<int64_t> strides);

  uint32_t getNumDim();
  uint64_t getSize();
};
