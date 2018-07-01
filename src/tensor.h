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

inline tensor2_t tensor2_transpose(tensor2 *t)
{
  tensor2_t tmp;
  tmp.sizes[0] = t->sizes[1];
  tmp.sizes[1] = t->sizes[0];
  tmp.strides[0] = t->strides[1];
  tmp.strides[1] = t->strides[0];
  return tmp;
}

// get size

inline uint64_t tensor2_get_size(tensor2_t *t);
inline uint64_t tensor3_get_size(tensor3_t *t);
inline uint64_t tensor4_get_size(tensor4_t *t);

// set sizes (will reset strides to block linear)

void tensor2_set_sizes(tensor2_t *t, uint32_t i, uint32_t j);
void tensor3_set_sizes(tensor3_t *t, uint32_t i, uint32_t j, uint32_t m);
void tensor4_set_sizes(tensor4_t *t, uint32_t i, uint32_t j, uint32_t m, uint32_t n);

// set linear strides

void tensor2_set_linear_strides(tensor2_t *t);
void tensor3_set_linear_strides(tensor3_t *t);
void tensor4_set_linear_strides(tensor4_t *t);

// index

inline uint64_t tensor2_index(tensor2_t *t, uint32_t i, uint32_t j);
inline uint64_t tensor3_index(tensor3_t *t, uint32_t i, uint32_t j, uint32_t m);
inline uint64_t tensor4_index(tensor4_t *t, uint32_t i, uint32_t j, uint32_t m, uint32_t n);

// allocate

void *tensor2_malloc_cuda(tensor2_t *t, uint8_t dsize);
void *tensor3_malloc_cuda(tensor3_t *t, uint8_t dsize);
void *tensor4_malloc_cuda(tensor4_t *t, uint8_t dsize);
void *tensor2_malloc(tensor2_t *t, uint8_t dsize);
void *tensor3_malloc(tensor3_t *t, uint8_t dsize);
void *tensor4_malloc(tensor4_t *t, uint8_t dsize);

// INLINE METHODS

// get size

inline uint64_t tensor2_get_size(tensor2_t *t)
{
  uint8_t maxdim = 0;
  uint8_t maxstride = t->strides[0];
  for (int i = 1; i < 2; i++) {
    if (t->strides[i] > maxstride) {
      maxdim = i;
      maxstride = t->strides[i];
    }
  }
  return t->sizes[maxdim] * t->strides[maxdim];
}

inline uint64_t tensor3_get_size(tensor3_t *t)
{
  uint8_t maxdim = 0;
  uint8_t maxstride = t->strides[0];
  for (int i = 1; i < 2; i++) {
    if (t->strides[i] > maxstride) {
      maxdim = i;
      maxstride = t->strides[i];
    }
  }
  return t->sizes[maxdim] * t->strides[maxdim];
}

inline uint64_t tensor4_get_size(tensor4_t *t)
{
  uint8_t maxdim = 0;
  uint8_t maxstride = t->strides[0];
  for (int i = 1; i < 2; i++) {
    if (t->strides[i] > maxstride) {
      maxdim = i;
      maxstride = t->strides[i];
    }
  }
  return t->sizes[maxdim] * t->strides[maxdim];
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
