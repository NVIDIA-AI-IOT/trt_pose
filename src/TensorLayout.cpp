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

TensorLayout::TensorLayout(std::vector<uint64_t> sizes) : TensorLayout(sizes, DefaultStrides(sizes))
{
}

TensorLayout::TensorLayout(std::vector<uint64_t> sizes, std::vector<int64_t> strides)
{
  this->sizes = sizes; 
  this->strides = strides;
}

uint64_t TensorLayout::GetSize()
{
  return sizes[0] * strides[0];
}

uint64_t TensorLayout::GetCount()
{
  uint64_t count = 1;
  for (auto &s : sizes) {
    count *= s;
  }
  return count;
}
