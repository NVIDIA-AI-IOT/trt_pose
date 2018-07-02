#pragma once

#include "matrix.h"
#include "cublas_v2.h"

void matrix_add_nn_d(
  cublasHandle_t handle,
  float *a_data, matrix_t *a_mat,
  float *b_data, matrix_t *b_mat,
  float *c_data, matrix_t *c_mat,
  float alpha=1.0, float beta=1.0)
{
  cublasSgeam(
      handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      c_mat->rows, c_mat->cols,
      &alpha,
      a_data, a_mat->rows,
      &beta,
      b_data, b_mat->rows,
      c_data, c_mat->rows
  );
}

void matrix_add_tn_d(
  cublasHandle_t handle,
  float *a_data, matrix_t *a_mat,
  float *b_data, matrix_t *b_mat,
  float *c_data, matrix_t *c_mat,
  float alpha=1.0, float beta=1.0)
{
  cublasSgeam(
      handle,
      CUBLAS_OP_T, CUBLAS_OP_N,
      c_mat->rows, c_mat->cols,
      &alpha,
      a_data, a_mat->rows,
      &beta,
      b_data, b_mat->rows,
      c_data, c_mat->rows
  );
}

void matrix_add_nt_d(
  cublasHandle_t handle,
  float *a_data, matrix_t *a_mat,
  float *b_data, matrix_t *b_mat,
  float *c_data, matrix_t *c_mat,
  float alpha=1.0, float beta=1.0)
{
  cublasSgeam(
      handle,
      CUBLAS_OP_N, CUBLAS_OP_T,
      c_mat->rows, c_mat->cols,
      &alpha,
      a_data, a_mat->rows,
      &beta,
      b_data, b_mat->rows,
      c_data, c_mat->rows
  );
}

void matrix_add_tt_d(
  cublasHandle_t handle,
  float *a_data, matrix_t *a_mat,
  float *b_data, matrix_t *b_mat,
  float *c_data, matrix_t *c_mat,
  float alpha=1.0, float beta=1.0)
{
  cublasSgeam(
      handle,
      CUBLAS_OP_T, CUBLAS_OP_T,
      c_mat->rows, c_mat->cols,
      &alpha,
      a_data, a_mat->rows,
      &beta,
      b_data, b_mat->rows,
      c_data, c_mat->rows
  );
}

