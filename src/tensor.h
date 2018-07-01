#pragma once

#include <cstdint>
#include <cstring>

typedef struct tensor2 {
  uint32_t sizes[2];
  int64_t strides[2];
} tensor2_t;

typedef struct tensor3 {
  uint32_t sizes[3];
  int64_t strides[3];
} tensor3_t;

typedef struct tensor4 {
  uint32_t sizes[4];
  int64_t strides[4];
} tensor4_t;

// get size

inline uint64_t tensor2_get_size(tensor2_t *t)
{
  return t->sizes[0] * t->strides[0];
}

inline uint64_t tensor3_get_size(tensor3_t *t)
{
  return t->sizes[0] * t->strides[0];
}

inline uint64_t tensor4_get_size(tensor4_t *t)
{
  return t->sizes[0] * t->strides[0];
}

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

// index

inline uint64_t tensor2_index(tensor2_t *t, uint32_t i, uint32_t j)
{
  return t->strides[0] * i + t->strides[1] * j;
}

inline uint64_t tensor3_index(tensor3_t *t, uint32_t i, uint32_t j, uint32_t m)
{
  return t->strides[0] * i + t->strides[1] * j + t->strides[2] * m;
}

inline uint64_t tensor4_index(tensor4_t *t, uint32_t i, uint32_t j, uint32_t m, uint32_t n)
{
  return t->strides[0] * i + t->strides[1] * j + t->strides[2] * m + t->strides[3] * n;
}
