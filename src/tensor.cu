#include "tensor.h"
#include "tensor.cuh"

void *tensor2_malloc_cuda(tensor2_t *t, uint8_t dsize)
{
  void *p;
  cudaMalloc(&p, dsize * tensor2_get_size(t));
  return p;
}

void *tensor3_malloc_cuda(tensor3_t *t, uint8_t dsize)
{
  void *p;
  cudaMalloc(&p, dsize * tensor3_get_size(t));
  return p;
}

void *tensor4_malloc_cuda(tensor4_t *t, uint8_t dsize)
{
  void *p;
  cudaMalloc(&p, dsize * tensor4_get_size(t));
  return p;
}

void *tensor2_malloc(tensor2_t *t, uint8_t dsize)
{
  return malloc(dsize * tensor2_get_size(t));
}

void *tensor3_malloc(tensor3_t *t, uint8_t dsize)
{
  return malloc(dsize * tensor3_get_size(t));
}

void *tensor4_malloc(tensor4_t *t, uint8_t dsize)
{
  return malloc(dsize * tensor4_get_size(t));
}
