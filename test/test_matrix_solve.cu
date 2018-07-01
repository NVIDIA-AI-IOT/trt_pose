#include "gtest/gtest.h"

#include "test_utils.h"

#include "cuda_runtime.h"

#include "../src/matrix.h"
#include "../src/matrix_fill.h"
#include "../src/matrix_solve.h"

TEST(matrix_solve, Inverse)
{
  matrix_t a, b;
  matrix_set_shape(&a, 2, 2);
  matrix_set_shape(&b, 2, 2);
  float *ah, *bh, *ad, *bd;

  matrix_malloc_h(&a, &ah);
  matrix_malloc_d(&a, &ad);

  matrix_malloc_h(&b, &bh);
  matrix_malloc_d(&b, &bd);

  // create source matrix
  float ah_T[2 * 2] = {
    2, -1,
    -1, 5
  };

  float bh_true_T[2 * 2] = {
    0.55555555, 0.11111111,
    0.11111111, 0.22222222
  };

  matrix_copy_h2h_transpose(&a, ah_T, ah);
  matrix_copy_h2d(&a, ah_T, ad);

  // create identity matrix (will hold inverse)
  matrix_fill_identity_d(&b, bd);

  cusolverDnHandle_t handle;
  cusolverDnCreate(&handle);

  int workspace_size = matrix_solve_c_workspace_size(handle, ad, &a);
  float *workspace;
  cudaMalloc(&workspace, sizeof(float) * workspace_size);

  matrix_solve_c(handle, ad, &a, bd, &b, workspace, workspace_size);

  matrix_copy_d2h(&b, bd, bh);

  AllFloatEqual(bh_true_T, bh, matrix_size(&b));

  free(ah);
  free(bh);
  cudaFree(ad);
  cudaFree(bd);
  cudaFree(workspace);
  cusolverDnDestroy(handle);
}

int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
