#pragma once

#include "cusolverDn.h"
#include "matrix.h"

int matrix_solve_workspace_size(
    cusolverDnHandle_t handle,
    float *a_data, matrix_t *a_mat)
{
  int workspace_size;
  cusolverDnSpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER,
      a_mat->rows, a_data, a_mat->rows, &workspace_size); 
  return workspace_size;
};

// must populate b_mat with identity matrix
int matrix_solve_c(
    cusolverDnHandle_t handle,
    float *a_data, matrix_t *a_mat,
    float *b_data, matrix_t *b_mat, 
    float *workspace, int workspace_size)
{
  // cholesky factorization
  cusolverDnSpotrf( handle, CUBLAS_FILL_MODE_LOWER, a_mat->rows, a_data,
    a_mat->rows, workspace, workspace_size, nullptr); 

  // linear solve
  cusolverDnSpotrs( handle, CUBLAS_FILL_MODE_LOWER, a_mat->rows, b_mat->cols,
      a_data, a_mat->rows, b_data, b_mat->rows, nullptr);

  return 0;
}
