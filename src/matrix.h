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

static inline float matrix_at(matrix_t *self, int row, int col)
{
  return self->data[IDX_2D(row, col, self->cols)];
}

static inline float * matrix_at_mutable(matrix_t *self, int row, int col)
{
  return &self->data[IDX_2D(row, col, self->cols)];
}

static inline float cmatrix_at(cmatrix_t *self, int row, int col)
{
  return self->data[IDX_2D_colmajor(row, col, self->rows)];
}

#ifdef __cplusplus
}
#endif

#endif
