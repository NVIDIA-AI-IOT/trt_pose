#ifndef MATRIX_INDEX_H
#define MATRIX_INDEX_H

#include "matrix.h"

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

#endif
