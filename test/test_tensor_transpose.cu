#include "gtest/gtest.h"

#include "cuda_runtime.h"

#include "../src/tensor.h"
#include "../src/tensor_transpose.h"

#include "test_utils.h"

TEST(tensor2_transpose, ValidSize) {
  tensor2_t a;
  tensor2_set_sizes(&a, 3, 2);
  tensor2_t b = tensor2_transpose(&a);
  ASSERT_EQ(6, tensor2_get_size(&b)); 
};

TEST(tensor2_transpose, SizesAndStrides) {
  tensor2_t a, b;
  tensor2_set_sizes(&a, 4, 2);
  b = tensor2_transpose(&a);
  ASSERT_EQ(b.sizes[0], a.sizes[1]);
  ASSERT_EQ(b.sizes[1], a.sizes[0]);
  ASSERT_EQ(b.strides[0], 4);
  ASSERT_EQ(b.strides[1], 1);
}

TEST(tensor2_transpose_data, ValidData) {
  cublasHandle_t handle;
  cublasCreate_v2(&handle);
  tensor2_t a, b;
  tensor2_set_sizes(&a, 4, 2);
  b = tensor2_transpose(&a);

  float aDh[4 * 2] = {
    0, 1,
    2, 3,
    4, 5,
    6, 7
  };

  float aDh_true[4 * 2] = {
    0, 1,
    2, 3,
    4, 5,
    6, 7
  };

  float bDh_true[2 * 4] = {
    0, 2, 4, 6,
    1, 3, 5, 7
  };
  float bDh[2 * 4];

  float *aD, *bD;
  cudaMalloc(&aD, sizeof(aDh));
  cudaMalloc(&bD, sizeof(bDh));
  cudaMemcpy(aD, aDh, sizeof(aDh), cudaMemcpyHostToDevice);
  cudaMemcpy(bD, bDh, sizeof(bDh), cudaMemcpyHostToDevice);

  tensor2_transpose_data(handle, aD, &a, bD, &b);

  cudaMemcpy(bDh, bD, sizeof(bDh), cudaMemcpyDeviceToHost);
  AllFloatEqual(bDh, bDh_true, 4 * 2);

  tensor2_transpose_data(handle, bD, &b, aD, &a);

  cudaMemcpy(aDh, aD, sizeof(aDh), cudaMemcpyDeviceToHost);
  AllFloatEqual(aDh, aDh_true, 4 * 2);

  cudaFree(aD);
  cudaFree(bD);
  cublasDestroy_v2(handle);
}

int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
