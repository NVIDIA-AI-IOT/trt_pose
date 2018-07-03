#include "gauss_newton.h"

#include "matrix_multiply.h"
#include "matrix_index.cuh"
#include "matrix_fill.h"
#include "matrix_solve.h"
#include "matrix_copy.h"


void gauss_newton_step(
    cublasHandle_t cublasHandle,
    cusolverDnHandle_t cusolverHandle,
    float *residual_data, matrix_t *residual_mat,
    float *jacobian_data, matrix_t *jacobian_mat,
    float *param_data, matrix_t *param_mat, float *workspace, int workspace_size, cudaStream_t streamId)
{
  // set stream for ops
  cublasSetStream_v2(cublasHandle, streamId);
  cusolverDnSetStream(cusolverHandle, streamId);

  // create temporary matrix
  matrix_t tmp_4x4_mat, tmp_4xNN_mat;
  matrix_set_shape(&tmp_4x4_mat, jacobian_mat->cols, jacobian_mat->cols);
  matrix_set_shape(&tmp_4xNN_mat, jacobian_mat->cols, jacobian_mat->rows);

  // assign matrices data to workspace locations
  float *tmp_4x4_data_0, *tmp_4x4_data_1, *tmp_4xNN_data, *solve_workspace;
  tmp_4x4_data_0 = workspace;
  tmp_4x4_data_1 = tmp_4x4_data_0 + matrix_size(&tmp_4x4_mat);
  tmp_4xNN_data = tmp_4x4_data_1 + matrix_size(&tmp_4x4_mat);
  solve_workspace = tmp_4xNN_data + matrix_size(&tmp_4xNN_mat);

  // compute J^T * J
  matrix_multiply_tn_c(cublasHandle, jacobian_data, jacobian_mat, jacobian_data, jacobian_mat, tmp_4x4_data_0);

  // compute (J^T * J) ^ -1
  //   create workspace
  int solve_workspace_size = matrix_solve_c_workspace_size(cusolverHandle, tmp_4x4_data_0, &tmp_4x4_mat); // we over-shoot

  //   initialize solution with identity
  matrix_fill_identity_d(&tmp_4x4_mat, tmp_4x4_data_1);
  //matrix_copy_d2d(&tmp_4x4_mat, identity_4x4_data, tmp_4x4_data_1);
  //matrix_copy_h2d_async(&tmp_4x4_mat, identity_4x4_data, tmp_4x4_data_1, streamId);

  //   solve
  matrix_solve_c(cusolverHandle, tmp_4x4_data_0, &tmp_4x4_mat, tmp_4x4_data_1, &tmp_4x4_mat, solve_workspace, solve_workspace_size);

  // compute ((J^T * J)^-1 * J^T)
  matrix_multiply_nt_c(cublasHandle, tmp_4x4_data_1, &tmp_4x4_mat, jacobian_data, jacobian_mat, tmp_4xNN_data);

  // compute param - (^) * residual
  matrix_multiply_nn_c(cublasHandle, tmp_4xNN_data, &tmp_4xNN_mat, residual_data, residual_mat, param_data, -1.0f, 1.0f);

  // reset stream
  cublasSetStream_v2(cublasHandle, NULL);
  cusolverDnSetStream(cusolverHandle, NULL);
}
