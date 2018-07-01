#include "matrix_fill.h"

void matrix_fill_identity_d(matrix_t *m, float *data);
__global__ void matrix_fill_identity_d_kernel(float *data, int N);

void matrix_fill_identity_d(matrix_t *m, float *data)
{
  dim3 blockDim = { 8, 8 };
  dim3 gridDim = { m->rows / 8 + 1, m->rows / 8 + 1 };
  matrix_fill_identity_d_kernel<<<gridDim, blockDim>>>(data, m->rows);
}

void matrix_fill_identity_d(matrix_t *m, float *data, cudaStream_t stream)
{
  dim3 blockDim = { 8, 8 };
  dim3 gridDim = { m->rows / 8 + 1, m->rows / 8 + 1 };
  matrix_fill_identity_d_kernel<<<gridDim, blockDim, 0, stream>>>(data, m->rows);
}

__global__ void matrix_fill_identity_d_kernel(float *data, int N)
{
  uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= N || j >= N) {
    return;
  }

  if (i == j) {
    data[N * j + i] = 1.0f;
  } else {
    data[N * j + i] = 0.0f;
  }
}
