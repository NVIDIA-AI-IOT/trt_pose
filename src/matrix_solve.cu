#include "matrix_solve.h"

int matrix_solve_c_workspace_size(
    cusolverDnHandle_t handle,
    float *a_data, matrix_t *a_mat)
{
  int workspace_size;
  cusolverDnSpotrf_bufferSize(handle, CUBLAS_FILL_MODE_UPPER,
      a_mat->rows, a_data, a_mat->rows, &workspace_size); 
  return workspace_size;
};

// must populate b_mat with identity matrix
// matrix must be symmetric positive-definite
int matrix_solve_c(
    cusolverDnHandle_t handle,
    float *a_data, matrix_t *a_mat,
    float *b_data, matrix_t *b_mat, 
    float *workspace, int workspace_size)
{
  int *info;
  cudaMalloc(&info, sizeof(int));

  // cholesky factorization
  cusolverDnSpotrf(handle, CUBLAS_FILL_MODE_UPPER, a_mat->rows, a_data,
    a_mat->rows, workspace, workspace_size, info); 

  // linear solve
  cusolverDnSpotrs(handle, CUBLAS_FILL_MODE_UPPER, a_mat->rows, b_mat->cols,
      a_data, a_mat->rows, b_data, b_mat->rows, info);

  cudaFree(info);
  return 0;
}
