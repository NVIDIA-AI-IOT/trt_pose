#include "gtest/gtest.h"

#include "test_utils.h"

#include "cuda_runtime.h"

#include "../src/matrix.h"
#include "../src/matrix_fill.h"
#include "../src/matrix_copy.h"
#include "../src/matrix_peak_threshold.h"

TEST(matrix_count_nonzero_h, Valid)
{
  matrix_t m;
  matrix_set_shape(&m, 3, 3);

  int count;
  uint8_t data0[] = {
    1, 0, 0,
    0, 1, 0,
    0, 1, 0
  };
  count = matrix_count_nonzero_h(&m, data0);
  ASSERT_EQ(3, count);

  uint8_t data1[] = {
    1, 0, 1,
    0, 1, 0,
    0, 1, 0
  };
  count = matrix_count_nonzero_h(&m, data1);
  ASSERT_EQ(4, count);

  uint8_t data2[] = {
    1, 0, 1,
    0, 1, 0,
    1, 1, 0
  };
  count = matrix_count_nonzero_h(&m, data2);
  ASSERT_EQ(5, count);

  uint8_t data3[] = {
    0, 0, 0,
    0, 0, 0,
    0, 0, 0
  };
  count = matrix_count_nonzero_h(&m, data3);
  ASSERT_EQ(0, count);

}

TEST(matrix_index_nonzero_h, Valid)
{
  matrix_t m;
  matrix_set_shape(&m, 3, 3);

  uint8_t data0[] = {
    1, 0, 0,
    0, 1, 0,
    0, 1, 0
  };
  uint64_t index0[] = { 0, 4, 7 };
  uint64_t index0_est[3];
  matrix_index_nonzero_h(&m, data0, index0_est);
  AllEqual(index0, index0_est, 3);

  uint8_t data1[] = {
    1, 0, 1,
    0, 1, 0,
    0, 1, 0
  };
  uint64_t index1[] = { 0, 2, 4, 7 };
  uint64_t index1_est[4];
  matrix_index_nonzero_h(&m, data1, index1_est);
  AllEqual(index1, index1_est, 4);

  uint8_t data2[] = {
    1, 0, 1,
    0, 1, 0,
    1, 1, 0
  };
  uint64_t index2[] = { 0, 2, 4, 6, 7 };
  uint64_t index2_est[5];
  matrix_index_nonzero_h(&m, data2, index2_est);
  AllEqual(index2, index2_est, 5);

}

TEST(matrix_peak_threshold_mask_d, Valid)
{
  matrix_t m;
  matrix_set_shape(&m, 4, 4);

  float data_h[4 * 4] = {
    0, 0, 0, 0,
    1, 2, 1, 0,
    0, 0, 0, 0,
    0, 0, 2, 0,
  };

  uint8_t mask_h[4 * 4];
  uint8_t mask_h_true[4 * 4] = {
    0, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 0, 0,
    0, 0, 1, 0
  };

  uint8_t *mask_d;
  float *data_d;
  matrix_malloc_d(&m, &mask_d);
  matrix_malloc_d(&m, &data_d);

  matrix_copy_h2d(&m, data_h, data_d);
  matrix_peak_threshold_mask_d(&m, data_d, mask_d, 1.0f);
  matrix_copy_d2h(&m, mask_d, mask_h);

  AllEqual(mask_h_true, mask_h, matrix_size(&m));

  cudaFree(mask_d);
  cudaFree(data_d);
}

#ifndef EXCLUDE_MAIN
int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif
