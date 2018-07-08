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

typedef struct imatrix {
  int *data;
  int rows;
  int cols;
} imatrix_t;

static inline float matrix_at(matrix_t *self, int row, int col)
{
  return self->data[IDX_2D(row, col, self->cols)];
}

static inline float * matrix_at_mutable(matrix_t *self, int row, int col)
{
  return &self->data[IDX_2D(row, col, self->cols)];
}

static inline void matrix_set_shape(matrix_t *self, int rows, int cols)
{
  self->rows = rows;
  self->cols = cols;
}

static inline int imatrix_at(imatrix_t *self, int row, int col)
{
  return self->data[IDX_2D(row, col, self->cols)];
}

static inline int * imatrix_at_mutable(imatrix_t *self, int row, int col)
{
  return &self->data[IDX_2D(row, col, self->cols)];
}

static inline void imatrix_set_shape(matrix_t *self, int rows, int cols)
{
  self->rows = rows;
  self->cols = cols;
}

#ifdef __cplusplus
}
#endif

#endif
