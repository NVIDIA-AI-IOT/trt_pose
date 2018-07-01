#include "tensor.h"

// set sizes

void tensor2_set_sizes(tensor2_t *t, uint32_t i, uint32_t j)
{
  t->sizes[0] = i;
  t->sizes[1] = j;
}

void tensor3_set_sizes(tensor3_t *t, uint32_t i, uint32_t j, uint32_t m)
{
  t->sizes[0] = i;
  t->sizes[1] = j;
  t->sizes[2] = m;
}

void tensor4_set_sizes(tensor4_t *t, uint32_t i, uint32_t j, uint32_t m, uint32_t n)
{
  t->sizes[0] = i;
  t->sizes[1] = j;
  t->sizes[2] = m;
  t->sizes[3] = n;
}

// set linear strides

void tensor2_set_linear_strides(tensor2_t *t)
{
  t->strides[1] = 1;
  t->strides[0] = t->sizes[1];
}

void tensor3_set_linear_strides(tensor3_t *t)
{
  t->strides[2] = 1;
  t->strides[1] = t->sizes[2];
  t->strides[0] = t->sizes[1] * t->strides[1];
}

void tensor4_set_linear_strides(tensor4_t *t)
{
  t->strides[3] = 1;
  t->strides[2] = t->sizes[3];
  t->strides[1] = t->sizes[2] * t->strides[2];
  t->strides[0] = t->sizes[1] * t->strides[1];
}

