#pragma once

#include <cstdint>

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
