#include "gtest/gtest.h"

#include "test_utils.h"

#include "cuda_runtime.h"

#include "../src/matrix.h"
#include "../src/matrix_fill.h"
#include "../src/matrix_copy.h"
#include "../src/matrix_peak_threshold.h"
#include "../src/cmap_peaks.h"

TEST(cmap_peaks, SingleChannel)
{
  matrix_t cmap_mat;
  matrix_set_shape(&cmap_mat, 4, 4);

  float cmap_data_h[4 * 4] = {
    0, 0, 0, 0,
    1, 2, 1, 0,
    0, 0, 0, 0,
    0, 0, 2, 0,
  };

  float *cmap_data_d;
  matrix_malloc_d(&cmap_mat, &cmap_data_d);
  matrix_copy_h2d(&cmap_mat, cmap_data_h, cmap_data_d);

  const int max_count = 100;
  const int cmap_channels = 1;

  int counts_h[cmap_channels];
  int peaks_h[cmap_channels * max_count];

  int *peaks_ptr_h[1];
  peaks_ptr_h[0] = &peaks_h[0];

  void *workspace_h, *workspace_d;

  cudaMallocHost(&workspace_h, cmap_peaks_workspace_h_size(cmap_channels));
  cudaMalloc(&workspace_d, cmap_peaks_workspace_d_size(cmap_channels, max_count));

  cmap_peaks(cmap_data_d, &cmap_mat, cmap_channels, 1.0f, counts_h, peaks_ptr_h, max_count, workspace_h, workspace_d);

  ASSERT_EQ(2, counts_h[0]);
  ASSERT_EQ(true, (peaks_h[0] == 5 || peaks_h[1] == 5) && (peaks_h[0] == 14 || peaks_h[1] == 14));

  cudaFree(cmap_data_d);
  cudaFree(workspace_h);
  cudaFree(workspace_d);
}

TEST(cmap_peaks, MultiChannel)
{
  matrix_t cmap_mat;
  matrix_set_shape(&cmap_mat, 4, 4);
  const int max_count = 100;
  const int cmap_channels = 2;

  float cmap_data_h[cmap_channels * 4 * 4] = {
    0, 0, 0, 0,
    1, 2, 1, 0,
    0, 0, 0, 0,
    0, 0, 2, 0,
    0, 0, 0, 0,
    1, 2, 1, 0,
    0, 0, 0, 0,
    0, 0, 2, 0,
  };

  float *cmap_data_d;
  cudaMalloc(&cmap_data_d, sizeof(float) * cmap_channels * matrix_size(&cmap_mat));
  cudaMemcpy(cmap_data_d, cmap_data_h, sizeof(cmap_data_h), cudaMemcpyHostToDevice);


  int counts_h[cmap_channels];
  int peaks_h[cmap_channels * max_count];

  int *peaks_ptr_h[2];
  peaks_ptr_h[0] = &peaks_h[0 * max_count];
  peaks_ptr_h[1] = &peaks_h[1 * max_count];

  void *workspace_h, *workspace_d;

  cudaMallocHost(&workspace_h, cmap_peaks_workspace_h_size(cmap_channels));
  cudaMalloc(&workspace_d, cmap_peaks_workspace_d_size(cmap_channels, max_count));

  cmap_peaks(cmap_data_d, &cmap_mat, cmap_channels, 1.0f, counts_h, peaks_ptr_h, max_count, workspace_h, workspace_d);

  ASSERT_EQ(2, counts_h[0]);
  ASSERT_EQ(2, counts_h[1]);
  ASSERT_EQ(true, (peaks_ptr_h[0][0] == 5 || peaks_ptr_h[0][1] == 5) && (peaks_ptr_h[0][0] == 14 || peaks_ptr_h[0][1] == 14));
  ASSERT_EQ(true, (peaks_ptr_h[1][0] == 5 || peaks_ptr_h[1][1] == 5) && (peaks_ptr_h[1][0] == 14 || peaks_ptr_h[1][1] == 14));

  cudaFree(cmap_data_d);
  cudaFree(workspace_h);
  cudaFree(workspace_d);
}

#ifndef EXCLUDE_MAIN
int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif
