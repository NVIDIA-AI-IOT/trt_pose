#pragma once

#include "matrix.h"
#include "matrix_index.h"

// copy

template<typename T>
inline void matrix_copy_h2d(matrix_t *m, T *src, T *dst)
{
  cudaMemcpy(dst, src, sizeof(T) * matrix_size(m), cudaMemcpyHostToDevice);
}

template<typename T>
inline void matrix_copy_d2h(matrix_t *m, T *src, T *dst)
{
  cudaMemcpy(dst, src, sizeof(T) * matrix_size(m), cudaMemcpyDeviceToHost);
}

template<typename T>
inline void matrix_copy_h2h_transpose(matrix_t *m, const T *a, T *b)
{
  for (uint32_t i = 0; i < m->rows; i++) {
    for (uint32_t j = 0; j < m->cols; j++) {
      b[matrix_index_c(m, i, j)] = a[matrix_index_r(m, i, j)];
    }
  }
}

template<typename T>
inline void matrix_copy_h2h(matrix_t *m, const T *a, T *b)
{
  memcpy(b, a, sizeof(T) * matrix_size(m));
}

template<typename T>
inline void matrix_copy_d2d(matrix_t *m, const T *a, T *b)
{
  cudaMemcpy(b, a, sizeof(T) * matrix_size(m), cudaMemcpyDeviceToDevice);
}
