#include "matrix_index.h"
#include "matrix_peak_threshold.h"

__global__ void matrix_peak_threshold_atomic_d_kernel(matrix_t m, float *data, float threshold, int *count, int *peaks, int max_count)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  // out of bounds
  if (i >= m.rows || j >= m.cols) {
    return; // pixel out of bounds
  }

  int idx = matrix_index_r(&m, i, j);
  float d = data[idx];

  if ((d < threshold) ||
      (i - 1 >= 0 && data[matrix_index_r(&m, i - 1, j)] > d) ||
      (j - 1 >= 0 && data[matrix_index_r(&m, i, j - 1)] > d) ||
      (i + 1 < m.rows && data[matrix_index_r(&m, i + 1, j)] > d) ||
      (j + 1 < m.cols && data[matrix_index_r(&m, i, j + 1)] > d))
  {
    return; // below threshold or higher neighbor
  }

  int m_count = atomicAdd(count, 1);
  if (m_count < max_count) {
    peaks[m_count] = idx;
  }
}

int matrix_peak_threshold_atomic_d(matrix_t *m, float *data, float threshold, int *count, int *peaks, int max_count, cudaStream_t streamId)
{
   dim3 blockDim = { 8, 8 };
  dim3 gridDim = { m->rows / 8 + 1, m->cols / 8 + 1 };
  matrix_peak_threshold_atomic_d_kernel<<<gridDim, blockDim, 0, streamId>>>(*m, data, threshold, count, peaks, max_count);
  return 0;
}

template<typename T>
__global__ void matrix_peak_threshold_mask_d_kernel(matrix_t m, T *data, uint8_t *mask, T threshold)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  // out of bounds
  if (i >= m.rows || j >= m.cols) {
    return;
  }

  int idx = matrix_index_r(&m, i, j);
  T d = data[idx];

  if (d < threshold) {
    mask[idx] = 0;
    return;
  }

  // check if neighbor is greater
  if (i - 1 >= 0 && data[matrix_index_r(&m, i - 1, j)] > d) {
    mask[idx] = 0;
    return;
  }

  if (j - 1 >= 0 && data[matrix_index_r(&m, i, j - 1)] > d) {
    mask[idx] = 0;
    return;
  }

  if (i + 1 < m.rows && data[matrix_index_r(&m, i + 1, j)] > d) {
    mask[idx] = 0;
    return;
  }

  if (j + 1 < m.cols && data[matrix_index_r(&m, i, j + 1)] > d) {
    mask[idx] = 0;
    return;
  }

  mask[idx] = 1;
}

template<typename T>
int matrix_peak_threshold_mask_d(matrix_t *m, T *data, uint8_t *mask, T threshold, cudaStream_t streamId)
{
  dim3 blockDim = { 8, 8 };
  dim3 gridDim = { m->rows / 8 + 1, m->cols / 8 + 1 };
  matrix_peak_threshold_mask_d_kernel<<<gridDim, blockDim, 0, streamId>>>(*m, data, mask, threshold);
  return 0;
}

// explicit instantiations

template __global__ void matrix_peak_threshold_mask_d_kernel(matrix_t m, float *data, uint8_t *mask, float threshold);
template int matrix_peak_threshold_mask_d(matrix_t *m, float *data, uint8_t *mask, float threshold, cudaStream_t streamId);
