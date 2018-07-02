#pragma once

#include <cstdint>
#include <cstring>
#include "cuda_runtime.h"

// matrix stored in column-major order
typedef struct matrix {
  uint32_t rows;
  uint32_t cols;
} matrix_t;

inline void matrix_set_shape(matrix_t *m, uint32_t rows, uint32_t cols);
inline uint64_t matrix_size(matrix_t *m);

// INLINE implementations
inline void matrix_set_shape(matrix_t *m, uint32_t rows, uint32_t cols)
{
  m->rows = rows;
  m->cols = cols;
}

inline uint64_t matrix_size(matrix_t *m)
{
  return m->rows * m->cols;
}

inline matrix_t matrix_transpose(matrix_t *m)
{
  matrix_t mT;
  matrix_set_shape(&mT, m->cols, m->rows);
  return mT;
};

// malloc

template<typename T>
inline void matrix_malloc_h(matrix_t *m, T **data)
{
  *data = (T*) malloc(sizeof(T) * matrix_size(m)); 
}

template<typename T>
inline void matrix_malloc_d(matrix_t *m, T **data)
{
  cudaMalloc(data, sizeof(T) * matrix_size(m));
}

