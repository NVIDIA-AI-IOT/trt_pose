#include "gtest/gtest.h"

#include "cuda_runtime.h"

#include "test_utils.h"

#include "../src/matrix.h"
#include "../src/matrix_index.cuh"
#include "../src/matrix_multiply.h"
#include "../src/matrix_copy.h"

TEST(matrix_multiply_nn_c, Valid)
{
  matrix_t a, b, c;
  matrix_set_shape(&a, 3, 2);
  matrix_set_shape(&b, 2, 3);
  matrix_set_shape(&c, 3, 3);

  // create data
  float ah[3 * 2], bh[3 * 2], ch[3 * 3], chTrue[3 * 3];
  float ahT[] = {
    0, 1,
    2, 3,
    4, 5
  };
  float bhT[] = {
    0, 1, 2,
    3, 4, 5
  };
  float chT[] = {
    3, 4, 5,
    9, 14, 19,
    15, 24, 33
  };
  matrix_copy_h2h_transpose(&a, ahT, ah);
  matrix_copy_h2h_transpose(&b, bhT, bh);
  matrix_copy_h2h_transpose(&c, chT, chTrue);

  // copy to device
  float *ad, *bd, *cd;
  cudaMalloc(&ad, sizeof(ah));
  cudaMalloc(&bd, sizeof(bh));
  cudaMalloc(&cd, sizeof(ch));
  cudaMemcpy(ad, ah, sizeof(ah), cudaMemcpyHostToDevice);
  cudaMemcpy(bd, bh, sizeof(bh), cudaMemcpyHostToDevice);

  cublasHandle_t handle;
  cublasCreate_v2(&handle);

  matrix_multiply_nn_c(handle, ad, &a, bd, &b, cd);

  cudaMemcpy(ch, cd, sizeof(ch), cudaMemcpyDeviceToHost);

  AllFloatEqual(ch, chTrue, matrix_size(&c));
  ASSERT_EQ(4, ch[matrix_index_c(&c, 0, 1)]);
  ASSERT_EQ(9, ch[matrix_index_c(&c, 1, 0)]);
  ASSERT_EQ(24, ch[matrix_index_c(&c, 2, 1)]);

  cublasDestroy_v2(handle);
  cudaFree(ad);
  cudaFree(bd);
  cudaFree(cd);
}

TEST(matrix_multiply_tt_c, Valid)
{
  matrix_t a, b, c;
  matrix_set_shape(&a, 3, 2);
  matrix_set_shape(&b, 2, 3);
  matrix_set_shape(&c, 2, 2);

  // create data
  float ah[3 * 2], bh[3 * 2], ch[2 * 2], chTrue[2 * 2];
  float ahT[] = {
    0, 1,
    2, 3,
    4, 5
  };
  float bhT[] = {
    0, 1, 2,
    3, 4, 5
  };
  float chT[] = {
    10, 28,
    13, 40
  };
  matrix_copy_h2h_transpose(&a, ahT, ah);
  matrix_copy_h2h_transpose(&b, bhT, bh);
  matrix_copy_h2h_transpose(&c, chT, chTrue);

  // copy to device
  float *ad, *bd, *cd;
  cudaMalloc(&ad, sizeof(ah));
  cudaMalloc(&bd, sizeof(bh));
  cudaMalloc(&cd, sizeof(ch));
  cudaMemcpy(ad, ah, sizeof(ah), cudaMemcpyHostToDevice);
  cudaMemcpy(bd, bh, sizeof(bh), cudaMemcpyHostToDevice);

  cublasHandle_t handle;
  cublasCreate_v2(&handle);

  matrix_multiply_tt_c(handle, ad, &a, bd, &b, cd);

  cudaMemcpy(ch, cd, sizeof(ch), cudaMemcpyDeviceToHost);

  AllFloatEqual(ch, chTrue, matrix_size(&c));
  ASSERT_EQ(10, ch[matrix_index_c(&c, 0, 0)]);
  ASSERT_EQ(28, ch[matrix_index_c(&c, 0, 1)]);
  ASSERT_EQ(13, ch[matrix_index_c(&c, 1, 0)]);
  ASSERT_EQ(40, ch[matrix_index_c(&c, 1, 1)]);

  cublasDestroy_v2(handle);
  cudaFree(ad);
  cudaFree(bd);
  cudaFree(cd);
}

TEST(matrix_multiply_nt_c, Valid)
{
  matrix_t a, b, c;
  matrix_set_shape(&a, 3, 2);
  matrix_set_shape(&b, 2, 3);
  matrix_set_shape(&c, 3, 3);

  // create data
  float ah[3 * 2], bh[3 * 2], ch[3 * 3], chTrue[3 * 3];
  float ahT[] = {
    0, 1,
    2, 3,
    4, 5
  };
  float bhT[] = {
    0, 1, 2,
    3, 4, 5
  };
  float chT[] = {
    1, 3, 5,
    3, 13, 23,
    5, 23, 41
  };
  matrix_copy_h2h_transpose(&a, ahT, ah);
  matrix_copy_h2h_transpose(&b, bhT, bh);
  matrix_copy_h2h_transpose(&c, chT, chTrue);

  // copy to device
  float *ad, *bd, *cd;
  cudaMalloc(&ad, sizeof(ah));
  cudaMalloc(&bd, sizeof(bh));
  cudaMalloc(&cd, sizeof(ch));
  cudaMemcpy(ad, ah, sizeof(ah), cudaMemcpyHostToDevice);
  cudaMemcpy(bd, bh, sizeof(bh), cudaMemcpyHostToDevice);

  cublasHandle_t handle;
  cublasCreate_v2(&handle);

  matrix_multiply_nt_c(handle, ad, &a, ad, &a, cd);

  cudaMemcpy(ch, cd, sizeof(ch), cudaMemcpyDeviceToHost);

  AllFloatEqual(ch, chTrue, matrix_size(&c));
  ASSERT_EQ(1, ch[matrix_index_c(&c, 0, 0)]);
  ASSERT_EQ(3, ch[matrix_index_c(&c, 0, 1)]);
  ASSERT_EQ(3, ch[matrix_index_c(&c, 1, 0)]);
  ASSERT_EQ(13, ch[matrix_index_c(&c, 1, 1)]);

  cublasDestroy_v2(handle);
  cudaFree(ad);
  cudaFree(bd);
  cudaFree(cd);
}

TEST(matrix_multiply_tn_c, Valid)
{
  matrix_t a, b, c;
  matrix_set_shape(&a, 3, 2);
  matrix_set_shape(&b, 2, 3);
  matrix_set_shape(&c, 2, 2);

  // create data
  float ah[3 * 2], bh[3 * 2], ch[2 * 2], chTrue[2 * 2];
  float ahT[] = {
    0, 1,
    2, 3,
    4, 5
  };
  float bhT[] = {
    0, 1, 2,
    3, 4, 5
  };
  float chT[] = {
    20, 26, 
    26, 35
  };
  matrix_copy_h2h_transpose(&a, ahT, ah);
  matrix_copy_h2h_transpose(&b, bhT, bh);
  matrix_copy_h2h_transpose(&c, chT, chTrue);

  // copy to device
  float *ad, *bd, *cd;
  cudaMalloc(&ad, sizeof(ah));
  cudaMalloc(&bd, sizeof(bh));
  cudaMalloc(&cd, sizeof(ch));
  cudaMemcpy(ad, ah, sizeof(ah), cudaMemcpyHostToDevice);
  cudaMemcpy(bd, bh, sizeof(bh), cudaMemcpyHostToDevice);

  cublasHandle_t handle;
  cublasCreate_v2(&handle);

  matrix_multiply_tn_c(handle, ad, &a, ad, &a, cd);

  cudaMemcpy(ch, cd, sizeof(ch), cudaMemcpyDeviceToHost);

  AllFloatEqual(ch, chTrue, matrix_size(&c));
  ASSERT_EQ(20, ch[matrix_index_c(&c, 0, 0)]);
  ASSERT_EQ(26, ch[matrix_index_c(&c, 0, 1)]);
  ASSERT_EQ(26, ch[matrix_index_c(&c, 1, 0)]);
  ASSERT_EQ(35, ch[matrix_index_c(&c, 1, 1)]);

  cublasDestroy_v2(handle);
  cudaFree(ad);
  cudaFree(bd);
  cudaFree(cd);
}

#ifndef EXCLUDE_MAIN
int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif
