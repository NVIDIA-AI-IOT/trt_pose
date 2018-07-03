#pragma once

#include "cublas_v2.h"
#include "matrix.h"

inline int matrix_multiply_nn_c(
    cublasHandle_t handle,
    float *a_data, matrix_t *a_mat,
    float *b_data, matrix_t *b_mat,
    float *c_data, float alpha=1.0f, float beta=0.0f)
{
  cublasSgemm(
      handle, CUBLAS_OP_N, CUBLAS_OP_N,
      a_mat->rows, b_mat->cols, a_mat->cols,
      &alpha,
      a_data, a_mat->rows,
      b_data, b_mat->rows,
      &beta,
      c_data, a_mat->rows
  );
  return 0;
}

inline int matrix_multiply_nt_c(
    cublasHandle_t handle,
    float *a_data, matrix_t *a_mat,
    float *b_data, matrix_t *b_mat,
    float *c_data, float alpha=1.0f, float beta=0.0f)
{
  cublasSgemm(
      handle, CUBLAS_OP_N, CUBLAS_OP_T,
      a_mat->rows, b_mat->rows, a_mat->cols,
      &alpha,
      a_data, a_mat->rows,
      b_data, b_mat->rows,
      &beta,
      c_data, a_mat->rows
  );
  return 0;
}

inline int matrix_multiply_tn_c(
    cublasHandle_t handle,
    float *a_data, matrix_t *a_mat,
    float *b_data, matrix_t *b_mat,
    float *c_data, float alpha=1.0f, float beta=0.0f)
{
  cublasSgemm(
      handle, CUBLAS_OP_T, CUBLAS_OP_N,
      a_mat->cols, b_mat->cols, a_mat->rows,
      &alpha,
      a_data, a_mat->rows,
      b_data, b_mat->rows,
      &beta,
      c_data, a_mat->cols
  );
  return 0;
}

inline int matrix_multiply_tt_c(
    cublasHandle_t handle,
    float *a_data, matrix_t *a_mat,
    float *b_data, matrix_t *b_mat,
    float *c_data, float alpha=1.0f, float beta=0.0f)
{
  cublasSgemm(
      handle, CUBLAS_OP_T, CUBLAS_OP_T,
      a_mat->cols, b_mat->rows, a_mat->rows,
      &alpha,
      a_data, a_mat->rows,
      b_data, b_mat->rows,
      &beta,
      c_data, a_mat->cols
  );
  return 0;
}
