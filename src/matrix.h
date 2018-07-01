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
inline uint64_t matrix_index_c(matrix_t *m, uint32_t row, uint32_t col); // column major index
inline uint64_t matrix_index_r(matrix_t *m, uint32_t row, uint32_t col); // row major index
inline uint64_t matrix_size(matrix_t *m);

// INLINE implementations
inline void matrix_set_shape(matrix_t *m, uint32_t rows, uint32_t cols)
{
  m->rows = rows;
  m->cols = cols;
}

// column major index
inline uint64_t matrix_index_c(matrix_t *m, uint32_t row, uint32_t col)
{
  return m->rows * col + row;
}

// row major index
inline uint64_t matrix_index_r(matrix_t *m, uint32_t row, uint32_t col)
{
  return m->cols * row + col;
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

inline void matrix_malloc_h(matrix_t *m, float **data)
{
  *data = (float*) malloc(sizeof(float) * matrix_size(m)); 
}

inline void matrix_malloc_d(matrix_t *m, float **data)
{
  cudaMalloc(data, sizeof(float) * matrix_size(m));
}

// copy

inline void matrix_copy_h2d(matrix_t *m, float *src, float *dst)
{
  cudaMemcpy(dst, src, sizeof(float) * matrix_size(m), cudaMemcpyHostToDevice);
}

inline void matrix_copy_d2h(matrix_t *m, float *src, float *dst)
{
  cudaMemcpy(dst, src, sizeof(float) * matrix_size(m), cudaMemcpyDeviceToHost);
}

inline void matrix_copy_h2h_transpose(matrix_t *m, const float *a, float *b)
{
  for (uint32_t i = 0; i < m->rows; i++) {
    for (uint32_t j = 0; j < m->cols; j++) {
      b[matrix_index_c(m, i, j)] = a[matrix_index_r(m, i, j)];
    }
  }
}
