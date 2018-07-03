#include "gtest/gtest.h"

#include "cuda_runtime.h"

#include "test_utils.h"

#include "../src/matrix.h"
#include "../src/matrix_index.h"
#include "../src/matrix_add.h"
#include "../src/matrix_copy.h"

TEST(matrix_add_nn_c, Valid)
{
  matrix_t a, b, c;
  matrix_set_shape(&a, 3, 2);
  matrix_set_shape(&b, 3, 2);
  matrix_set_shape(&c, 3, 2);

  float ah[] = {
    1, 2,
    3, 4,
    5, 6
  };
  float bh[] = {
    1, 2,
    3, 4,
    5, 6
  };
  float ch[3*2];
  float ch_true[] = {
    2, 4,
    6, 8,
    10, 12
  };
  matrix_copy_h2h_transpose(&c, ch_true, ch);
  matrix_copy_h2h(&c, ch, ch_true);
  matrix_copy_h2h_transpose(&a, ah, ch);
  matrix_copy_h2h(&a, ch, ah);
  matrix_copy_h2h_transpose(&b, bh, ch);
  matrix_copy_h2h(&b, ch, bh);

  float *ad, *bd, *cd;
  matrix_malloc_d(&a, &ad);
  matrix_malloc_d(&b, &bd);
  matrix_malloc_d(&c, &cd);
  matrix_copy_h2d(&a, ah, ad);
  matrix_copy_h2d(&b, bh, bd);
  matrix_copy_h2d(&c, ch, cd);
  
  cublasHandle_t handle;
  cublasCreate_v2(&handle);

  matrix_add_nn_d(handle, ad, &a, bd, &b, cd, &c);

  matrix_copy_d2h(&c, cd, ch);

  AllFloatEqual(ch_true, ch, matrix_size(&c));

  cudaFree(ad);
  cudaFree(bd);
  cudaFree(cd);
  cublasDestroy_v2(handle);
}

TEST(matrix_add_tt_c, Valid)
{
  matrix_t a, b, c;
  matrix_set_shape(&a, 3, 2);
  matrix_set_shape(&b, 3, 2);
  matrix_set_shape(&c, 2, 3);

  float ah[] = {
    1, 2,
    3, 4,
    5, 6
  };
  float bh[] = {
    1, 2,
    3, 4,
    5, 6
  };
  float ch[3*2];
  float ch_true[] = {
    2, 6, 10,
    4, 8, 12
  };
  matrix_copy_h2h_transpose(&c, ch_true, ch);
  matrix_copy_h2h(&c, ch, ch_true);
  matrix_copy_h2h_transpose(&a, ah, ch);
  matrix_copy_h2h(&a, ch, ah);
  matrix_copy_h2h_transpose(&b, bh, ch);
  matrix_copy_h2h(&b, ch, bh);

  float *ad, *bd, *cd;
  matrix_malloc_d(&a, &ad);
  matrix_malloc_d(&b, &bd);
  matrix_malloc_d(&c, &cd);
  matrix_copy_h2d(&a, ah, ad);
  matrix_copy_h2d(&b, bh, bd);
  matrix_copy_h2d(&c, ch, cd);
  
  cublasHandle_t handle;
  cublasCreate_v2(&handle);

  matrix_add_tt_d(handle, ad, &a, bd, &b, cd, &c);

  matrix_copy_d2h(&c, cd, ch);

  AllFloatEqual(ch_true, ch, matrix_size(&c));

  cudaFree(ad);
  cudaFree(bd);
  cudaFree(cd);
  cublasDestroy_v2(handle);
}

TEST(matrix_add_nn_c_inplace, Valid)
{
  matrix_t a, b, c;
  matrix_set_shape(&a, 3, 2);
  matrix_set_shape(&b, 3, 2);
  matrix_set_shape(&c, 3, 2);

  float ah[] = {
    1, 2,
    3, 4,
    5, 6
  };

  float bh[] = {
    1, 2,
    3, 4,
    5, 6
  };

  float ch[3*2];
  float ch_true[] = {
    2, 4,
    6, 8,
    10, 12
  };
  matrix_copy_h2h_transpose(&c, ch_true, ch);
  matrix_copy_h2h(&c, ch, ch_true);
  matrix_copy_h2h_transpose(&a, ah, ch);
  matrix_copy_h2h(&a, ch, ah);
  matrix_copy_h2h_transpose(&b, bh, ch);
  matrix_copy_h2h(&b, ch, bh);

  float *ad, *bd, *cd;
  matrix_malloc_d(&a, &ad);
  matrix_malloc_d(&b, &bd);
  matrix_malloc_d(&c, &cd);
  matrix_copy_h2d(&a, ah, ad);
  matrix_copy_h2d(&b, bh, bd);
  matrix_copy_h2d(&c, ch, cd);
  
  cublasHandle_t handle;
  cublasCreate_v2(&handle);

  matrix_add_nn_d(handle, ad, &a, bd, &b, ad, &a);

  matrix_copy_d2h(&a, ad, ch);

  AllFloatEqual(ch_true, ch, matrix_size(&c));

  cudaFree(ad);
  cudaFree(bd);
  cudaFree(cd);
  cublasDestroy_v2(handle);
}

#ifndef EXCLUDE_MAIN
int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif
