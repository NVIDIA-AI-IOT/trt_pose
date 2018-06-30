#include "TensorLayout.h"

std::vector<int64_t> DefaultStrides(std::vector<uint64_t> sizes)
{
  if (sizes.size() == 0) {
    return {};
  }
  
  std::vector<int64_t> strides(sizes.size());
  strides[sizes.size() - 1] = 1;

  for (int i = sizes.size() - 2; i >= 0; i--) {
    strides[i] = sizes[i + 1] * strides[i + 1];
  }

  return strides;
}

TensorLayout::TensorLayout(std::vector<uint64_t> sizes)
{
  sizes = sizes;
}

TensorLayout::TensorLayout(std::vector<uint64_t> sizes, std::vector<int64_t> strides)
{
}

