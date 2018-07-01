#pragma once

#include <cstdint>

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
