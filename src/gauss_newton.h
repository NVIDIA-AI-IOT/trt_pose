#pragma once

#include "matrix.h"
#include "cublas_v2.h"
#include "cusolverDn.h"

inline int gauss_newton_step_workspace_size(matrix_t *jacobian_mat) {
  return sizeof(float) * (3 * (jacobian_mat->cols * jacobian_mat->cols) + jacobian_mat->cols * jacobian_mat->rows); // 3  4x4 matrices, 1 4x(N*N) matrix
};

void gauss_newton_step(
    cublasHandle_t cublasHandle,
    cusolverDnHandle_t cusolverHandle,
    float *residual_data, matrix_t *residual_mat,
    float *jacobian_data, matrix_t *jacobian_mat,
    float *param_data, matrix_t *param_mat, float *workspace, cudaStream_t streamId=NULL); 
