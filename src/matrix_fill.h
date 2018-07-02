#pragma once

#include "matrix.h"
#include "matrix_index.h"
#include "cuda_runtime.h"

// identity on host
template<typename T>
void matrix_fill_identity_h(matrix_t *m, T *data)
{
  for (uint32_t i = 0; i < m->rows; i++) {
    for (uint32_t j = 0; j < m->cols; j++) {
      uint64_t idx = matrix_index_c(m, i, j);
      if (i == j) {
        data[idx] = 1;
      } else {
        data[idx] = 0;
      }
    }
  }
}

// identity on device
template<typename T>
void matrix_fill_identity_d(matrix_t *m, T *data);

template<typename T>
void matrix_fill_identity_d(matrix_t *m, T *data, cudaStream_t stream);
