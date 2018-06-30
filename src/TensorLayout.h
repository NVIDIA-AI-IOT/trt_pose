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

  uint64_t GetSize();
  uint64_t GetCount(); // number of valid elements (exclude stride)
};
