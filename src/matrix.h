#ifndef MATRIX_H
#define MATRIX_H

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// row major matrix
typedef struct matrix {
  float *data;
  int rows;
  int cols;
} matrix_t;

// col major matrix
typedef struct cmatrix {
  float *data;
  int rows;
  int cols;
} cmatrix_t;

extern inline float matrix_at(matrix_t *self, int row, int col)
{
  return self->data[IDX_2D(row, col, self->cols)];
}

extern inline float cmatrix_at(cmatrix_t *self, int row, int col)
{
  return self->data[IDX_2D_colmajor(row, col, self->rows)];
}

#ifdef __cplusplus
}
#endif

#endif
