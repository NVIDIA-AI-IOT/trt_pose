#include "matrix_fill.h"

template<typename T>
void matrix_fill_identity_d(matrix_t *m, T *data);

template<typename T>
__global__ void matrix_fill_identity_d_kernel(T *data, int N);

template<typename T>
void matrix_fill_identity_d(matrix_t *m, T *data)
{
  dim3 blockDim = { 8, 8 };
  dim3 gridDim = { m->rows / 8 + 1, m->rows / 8 + 1 };
  matrix_fill_identity_d_kernel<<<gridDim, blockDim>>>(data, m->rows);
}

template<typename T>
void matrix_fill_identity_d(matrix_t *m, T *data, cudaStream_t stream)
{
  dim3 blockDim = { 8, 8 };
  dim3 gridDim = { m->rows / 8 + 1, m->rows / 8 + 1 };
  matrix_fill_identity_d_kernel<<<gridDim, blockDim, 0, stream>>>(data, m->rows);
}

template<typename T>
__global__ void matrix_fill_identity_d_kernel(T *data, int N)
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

// explicitly instantiate templates

template void matrix_fill_identity_d(matrix_t *m, float *data);
template void matrix_fill_identity_d(matrix_t *m, uint8_t *data);
template __global__ void matrix_fill_identity_d_kernel(float *, int);
template __global__ void matrix_fill_identity_d_kernel(uint8_t *, int);
