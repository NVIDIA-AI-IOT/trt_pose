#pragma once

#include "matrix.h"
#include "matrix_index.h"
#include "gauss_newton.h"

// cmap_data is row-major (as per tensorRt)
// residual, jacobian, and params are all column-major (as per cublas/cusolver)
void residual_jacobian_d(
    uint64_t idx, uint8_t N,
    float *cmap_data, matrix_t *cmap_mat,
    float *residual_data, matrix_t *residual_mat,
    float *jacobian_data, matrix_t *jacobian_mat,
    float *param_data, matrix_t *param_mat, cudaStream_t streamId=NULL);

int gaussian_fit_workspace_size(uint8_t N) {
  matrix_t jmat;
  matrix_set_shape(&jmat, N*N, 4);
  // residual matrix size + jacobian matrix size + gauss_newton_optimization workspace size
  return sizeof(float) * (jmat.rows + jmat.rows * jmat.cols) + gauss_newton_step_workspace_size(&jmat);
}

void gaussian_fit(
  cublasHandle_t cublasHandle,
  cusolverDnHandle_t cusolverHandle,
  int niters,
  uint64_t idx, uint8_t N,
  float *cmap_data, matrix_t *cmap_mat,
  float *param_data, matrix_t *param_mat,
  float *workspace, cudaStream_t streamId=NULL)
{
  matrix_t jacobian_mat, residual_mat;
  matrix_set_shape(&residual_mat, N * N, 1);
  matrix_set_shape(&jacobian_mat, N * N, 4);
  float *residual_data = workspace;
  float *jacobian_data = residual_data + matrix_size(&residual_mat);
  float *gauss_newton_workspace = jacobian_data + matrix_size(&jacobian_mat);

  for (int i = 0; i < niters; i++) {
    // compute residual and jacobian
    residual_jacobian_d(idx, N, cmap_data, cmap_mat, residual_data, &residual_mat, jacobian_data, &jacobian_mat, param_data, param_mat, streamId);
    // update parameters
    gauss_newton_step(cublasHandle, cusolverHandle, residual_data, &residual_mat, jacobian_data, &jacobian_mat, param_data, param_mat, gauss_newton_workspace,
        streamId);
  } 
}

