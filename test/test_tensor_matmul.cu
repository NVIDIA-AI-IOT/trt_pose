#include "gtest/gtest.h"

#include "../src/tensor.h"
#include "../src/tensor_matmul.h"

#include "test_utils.h"

TEST(tensor2_matmul, Valid) {
  cublasHandle_t handle;
  cublasCreate_v2(&handle);

  tensor2_t aT, bT, cT;
  tensor2_set_sizes(&aT, 2, 2);
  tensor2_set_sizes(&bT, 2, 2);
  tensor2_set_sizes(&cT, 2, 2);

  float aDh[2 * 2] = {
    1, 2, 
    3, 4
  };

  float bDh[2 * 2] = {
    2, 3,
    4, 5
  };

  float cDh[2 * 2];
  float cDh_true[2 * 2] = {
    10, 13,
    22, 29
  };

  float *aD, *bD, *cD;

  cudaMalloc(&aD, sizeof(float) * tensor2_get_size(&aT));
  cudaMalloc(&bD, sizeof(float) * tensor2_get_size(&bT));
  cudaMalloc(&cD, sizeof(float) * tensor2_get_size(&cT));

  cudaMemcpy(aD, aDh, sizeof(float) * tensor2_get_size(&aT), cudaMemcpyHostToDevice);
  cudaMemcpy(bD, bDh, sizeof(float) * tensor2_get_size(&bT), cudaMemcpyHostToDevice);

  tensor2_matmul(handle, CUBLAS_OP_N, CUBLAS_OP_N,
    aD, &aT, bD, &bT, cD, &cT);

  cudaMemcpy(cDh, cD, sizeof(float) * tensor2_get_size(&cT), cudaMemcpyDeviceToHost);

  ASSERT_EQ(cDh_true[tensor2_index(&cT, 0, 0)], cDh[tensor2_index(&cT, 0, 0)]);
  ASSERT_EQ(cDh_true[tensor2_index(&cT, 0, 1)], cDh[tensor2_index(&cT, 0, 1)]);
  ASSERT_EQ(cDh_true[tensor2_index(&cT, 1, 1)], cDh[tensor2_index(&cT, 1, 1)]);
  ASSERT_EQ(cDh_true[tensor2_index(&cT, 1, 0)], cDh[tensor2_index(&cT, 1, 0)]);

  cublasDestroy_v2(handle);
}

int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
