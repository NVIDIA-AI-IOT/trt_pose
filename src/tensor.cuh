#pragma once 

#include "tensor.h"

// index

__device__ inline uint64_t tensor2_index_cuda(tensor2_t *t, uint32_t i, uint32_t j)
{
  return t->strides[0] * i + t->strides[1] * j;
}

__device__ inline uint64_t tensor3_index_cuda(tensor3_t *t, uint32_t i, uint32_t j, uint32_t m)
{
  return t->strides[0] * i + t->strides[1] * j + t->strides[2] * m;
}

__device__ inline uint64_t tensor4_index_cuda(tensor4_t *t, uint32_t i, uint32_t j, uint32_t m, uint32_t n)
{
  return t->strides[0] * i + t->strides[1] * j + t->strides[2] * m + t->strides[3] * n;
}
