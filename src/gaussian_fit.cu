#include "gaussian_fit.h"

#include "matrix_multiply.h"
#include "matrix_index.cuh"
#include "matrix_fill.h"
#include "matrix_solve.h"
#include "matrix_copy.h"


// computes residual and jacobian of gaussian fit centered around index
template<typename T>
__global__ void residual_jacobian_d_kernel(
    uint64_t idx, uint8_t N,
    T *cmap_data, matrix_t cmap_mat,
    T *residual_data, matrix_t residual_mat,
    T *jacobian_data, matrix_t jacobian_mat,
    T *param_data, matrix_t param_mat)
{
  int i_offset = threadIdx.x - N / 2;
  int j_offset = threadIdx.y - N / 2;
  
  int i_peak = matrix_unravel_row_r(&cmap_mat, idx);
  int j_peak = matrix_unravel_col_r(&cmap_mat, idx);

  int i = i_peak + i_offset;
  int j = j_peak + j_offset;

  int residual_row = N * threadIdx.x + threadIdx.y;

  // set jacobian and resiudla to 0 if sample is out of bounds
  if (i < 0 || (uint32_t) i >= cmap_mat.rows || j < 0 || (uint32_t) j >= cmap_mat.cols)
  {
    residual_data[matrix_index_c(&residual_mat, residual_row, 0)] = 0;
    jacobian_data[matrix_index_c(&jacobian_mat, residual_row, 0)] = 0;
    jacobian_data[matrix_index_c(&jacobian_mat, residual_row, 1)] = 0;
    jacobian_data[matrix_index_c(&jacobian_mat, residual_row, 2)] = 0;
    jacobian_data[matrix_index_c(&jacobian_mat, residual_row, 3)] = 0;
    return;
  }

  // compute jacobian and jacobian
  T i_diff = i - param_data[0];
  T j_diff = j - param_data[0];
  T i_diff_2 = i_diff * i_diff;
  T j_diff_2 = j_diff * j_diff;
  T exp_val = exp(-(i_diff_2 + j_diff_2) / (2.0 * param_data[3]));
  T ij_coef = -param_data[2] * exp_val / param_data[3];

  residual_data[matrix_index_c(&residual_mat, residual_row, 0)] = cmap_data[matrix_index_r(&cmap_mat, i, j)] - param_data[2] * exp_val;

  jacobian_data[matrix_index_c(&jacobian_mat, residual_row, 0)] = ij_coef * i_diff;
  jacobian_data[matrix_index_c(&jacobian_mat, residual_row, 1)] = ij_coef * j_diff;
  jacobian_data[matrix_index_c(&jacobian_mat, residual_row, 2)] = -exp_val;
  jacobian_data[matrix_index_c(&jacobian_mat, residual_row, 3)] = ij_coef * (i_diff_2 + j_diff_2) / (2.0 * param_data[3]);

}

// computes residual and jacobian of gaussian fit centered around index
// residual mat should be (NxN)x1
// jacobian should be (NxN)x4
// param data should be 4x1
template<typename T>
void residual_jacobian_d(
    uint64_t idx, uint8_t N,
    T *cmap_data, matrix_t *cmap_mat,
    T *residual_data, matrix_t *residual_mat,
    T *jacobian_data, matrix_t *jacobian_mat,
    T *param_data, matrix_t *param_mat, cudaStream_t streamId)
{
  static const dim3 blockDim = { N, N }; // 3x3 pixel window used to appx
  residual_jacobian_d_kernel<<<1, blockDim, 0, streamId>>>(idx, N,
      cmap_data, *cmap_mat,
      residual_data, *residual_mat,
      jacobian_data, *jacobian_mat,
      param_data, *param_mat);
}

template __global__ void residual_jacobian_d_kernel(uint64_t, uint8_t, float *, matrix_t, float *, matrix_t, float*, matrix_t, float*, matrix_t);
template void residual_jacobian_d(uint64_t, uint8_t, float *, matrix_t *, float *, matrix_t *, float*, matrix_t *, float*, matrix_t *, cudaStream_t);
