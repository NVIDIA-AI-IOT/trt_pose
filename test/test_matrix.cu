#include "gtest/gtest.h"

#include "cuda_runtime.h"

#include "test_utils.h"

#include "../src/matrix.h"
#include "../src/matrix_index.cuh"
#include "../src/matrix_copy.h"

TEST(matrix_index, Valid)
{
  matrix_t m;
  matrix_set_shape(&m, 3, 5);

  // test column major indexing
  ASSERT_EQ(0, matrix_index_c(&m, 0, 0)); 
  ASSERT_EQ(1, matrix_index_c(&m, 1, 0)); 
  ASSERT_EQ(3, matrix_index_c(&m, 0, 1)); 
  ASSERT_EQ(4, matrix_index_c(&m, 1, 1)); 
  ASSERT_EQ(6, matrix_index_c(&m, 0, 2)); 

  // test row major indexing
  ASSERT_EQ(0, matrix_index_r(&m, 0, 0)); 
  ASSERT_EQ(5, matrix_index_r(&m, 1, 0)); 
  ASSERT_EQ(1, matrix_index_r(&m, 0, 1)); 
  ASSERT_EQ(6, matrix_index_r(&m, 1, 1)); 
  ASSERT_EQ(2, matrix_index_r(&m, 0, 2)); 
}

TEST(matrix_size, Valid)
{
  matrix_t m;
  matrix_set_shape(&m, 3, 5);
  ASSERT_EQ(15, matrix_size(&m));
}

TEST(matrix_copy_h2h_transpose, Valid)
{
  matrix_t m;
  matrix_set_shape(&m, 3, 2);
  float dT[] = {
    0, 1,
    2, 3,
    4, 5
  };
  float d[3 * 2];
  matrix_copy_h2h_transpose(&m, dT, d);
  ASSERT_EQ(0, d[0]);
  ASSERT_EQ(2, d[1]);
  ASSERT_EQ(4, d[2]);
  ASSERT_EQ(1, d[3]);
  ASSERT_EQ(0, d[matrix_index_c(&m, 0, 0)]);
  ASSERT_EQ(2, d[matrix_index_c(&m, 1, 0)]);
  ASSERT_EQ(5, d[matrix_index_c(&m, 2, 1)]);
}

TEST(matrix_unravel, Valid)
{
  matrix_t m;
  matrix_set_shape(&m, 3, 2);

  // col major
  ASSERT_EQ(0, matrix_unravel_col_c(&m, 0));
  ASSERT_EQ(0, matrix_unravel_row_c(&m, 0));
  
  ASSERT_EQ(0, matrix_unravel_col_c(&m, 1));
  ASSERT_EQ(1, matrix_unravel_row_c(&m, 1));

  ASSERT_EQ(1, matrix_unravel_col_c(&m, 3));
  ASSERT_EQ(0, matrix_unravel_row_c(&m, 3));

  ASSERT_EQ(1, matrix_unravel_col_c(&m, 5));
  ASSERT_EQ(2, matrix_unravel_row_c(&m, 5));

  // row major
  ASSERT_EQ(0, matrix_unravel_col_r(&m, 0));
  ASSERT_EQ(0, matrix_unravel_row_r(&m, 0));
  
  ASSERT_EQ(1, matrix_unravel_col_r(&m, 1));
  ASSERT_EQ(0, matrix_unravel_row_r(&m, 1));

  ASSERT_EQ(1, matrix_unravel_col_r(&m, 3));
  ASSERT_EQ(1, matrix_unravel_row_r(&m, 3));

  ASSERT_EQ(1, matrix_unravel_col_r(&m, 5));
  ASSERT_EQ(2, matrix_unravel_row_r(&m, 5));

}

TEST(matrix_copy_async, Valid)
{
  matrix_t m;
  matrix_set_shape(&m, 2, 2);
  float mh_init[] = {
    1, 2,
    3, 4
  };
  float mh[2*2];
  float *md; 
  matrix_malloc_d(&m, &md);

  cudaStream_t streamId;
  cudaStreamCreate(&streamId);

  matrix_copy_h2d_async(&m, mh_init, md, streamId);
  matrix_copy_d2h_async(&m, md, mh, streamId);

  cudaStreamSynchronize(streamId);

  AllFloatEqual(mh_init, mh, matrix_size(&m));

  cudaStreamDestroy(streamId);
  cudaFree(md);
}

#ifndef EXCLUDE_MAIN
int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif
