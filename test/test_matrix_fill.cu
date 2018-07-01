#include "gtest/gtest.h"

#include "test_utils.h"

#include "cuda_runtime.h"

#include "../src/matrix.h"
#include "../src/matrix_fill.h"

TEST(matrix_fill_identity_dh, Valid)
{
  matrix_t m;
  matrix_set_shape(&m, 3, 3);
  float *dh, *dd, *dd_h;

  dh = (float*) malloc(sizeof(float) * matrix_size(&m));
  dd_h = (float*) malloc(sizeof(float) * matrix_size(&m));
  cudaMalloc(&dd, sizeof(float) * matrix_size(&m));

  matrix_fill_identity_h(&m, dh);
  matrix_fill_identity_d(&m, dd);

  cudaMemcpy(dd_h, dd, sizeof(float) * matrix_size(&m), cudaMemcpyDeviceToHost);

  ASSERT_FLOAT_EQ(1, dh[matrix_index_c(&m, 0, 0)]);
  ASSERT_FLOAT_EQ(0, dh[matrix_index_c(&m, 0, 1)]);
  ASSERT_FLOAT_EQ(0, dh[matrix_index_c(&m, 0, 2)]);
  ASSERT_FLOAT_EQ(0, dh[matrix_index_c(&m, 1, 0)]);
  ASSERT_FLOAT_EQ(1, dh[matrix_index_c(&m, 1, 1)]);
  ASSERT_FLOAT_EQ(0, dh[matrix_index_c(&m, 1, 2)]);
  ASSERT_FLOAT_EQ(0, dh[matrix_index_c(&m, 2, 0)]);
  ASSERT_FLOAT_EQ(0, dh[matrix_index_c(&m, 2, 1)]);
  ASSERT_FLOAT_EQ(1, dh[matrix_index_c(&m, 2, 2)]);

  AllFloatEqual(dh, dd_h, matrix_size(&m));

  cudaFree(dd);
  free(dh);
  free(dd_h);
}

int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
