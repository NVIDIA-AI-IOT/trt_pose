#pragma once

#include "matrix_peak_threshold.h"

size_t cmap_peaks_workspace_h_size(int cmap_channels)
{
  return sizeof(cudaStream_t) * cmap_channels;
}

size_t cmap_peaks_workspace_d_size(int cmap_channels, int max_count)
{
  return cmap_channels * sizeof(int) // peak counts array
    + cmap_channels * max_count * sizeof(int); // peak index array
}

void cmap_peaks(float *cmap_data_d, matrix_t *cmap_mat_h, int cmap_channels, float threshold, int *counts_h, int **peaks_h, int max_count, void *workspace_h, void *workspace_d)
{
  cudaStream_t *streams;
  streams = (cudaStream_t *) workspace_h;
  int *counts_d;
  counts_d = (int *) workspace_d;
  cudaMemset(counts_d, 0, cmap_channels * sizeof(int)); // init counts to 0
  int *peaks_d;
  peaks_d = counts_d + cmap_channels;

  for (int i = 0; i < cmap_channels; i++) {
    cudaStreamCreate(&streams[i]);
    matrix_peak_threshold_atomic_d(cmap_mat_h, cmap_data_d + i * matrix_size(cmap_mat_h), threshold, counts_d + i, peaks_d + i * max_count, max_count, streams[i]);
  }

  for (int i = 0; i < cmap_channels; i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);

    cudaMemcpy(counts_h + i, counts_d + i, sizeof(int), cudaMemcpyDeviceToHost);
    if (counts_h[i] > max_count) {
      counts_h[i] = max_count;
    }
    cudaMemcpy(peaks_h[i], peaks_d + i * max_count, sizeof(int) * counts_h[i], cudaMemcpyDeviceToHost);
  }
}
