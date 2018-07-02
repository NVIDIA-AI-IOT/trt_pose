#pragma once

#include "matrix.h"
#include "cublas_v2.h"
#include "cusolverDn.h"

inline int update_param_workspace_size(uint8_t N) {
  return sizeof(float) * (3 * (4 * 4) + 4 * N * N); // 3  4x4 matrices, 1 4x(N*N) matrix
};

void update_param(
    cublasHandle_t cublasHandle,
    cusolverDnHandle_t cusolverHandle,
    uint64_t idx, uint8_t N,
    float *cmap_data, matrix_t *cmap_mat,
    float *residual_data, matrix_t *residual_mat,
    float *jacobian_data, matrix_t *jacobian_mat,
    float *param_data, matrix_t *param_mat, float *workspace, int workspace_size, cudaStream_t streamId=NULL); 

// cmap_data is row-major (as per tensorRt)
// residual, jacobian, and params are all column-major (as per cublas/cusolver)
template<typename T>
void residual_jacobian_d(
    uint64_t idx, uint8_t N,
    T *cmap_data, matrix_t *cmap_mat,
    T *residual_data, matrix_t *residual_mat,
    T *jacobian_data, matrix_t *jacobian_mat,
    T *param_data, matrix_t *param_mat, cudaStream_t streamId=NULL);
