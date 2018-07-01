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

  AllFloatEqual(cDh, cDh_true, 2 * 2);

  cublasDestroy_v2(handle);
  cudaFree(aD);
  cudaFree(bD);
  cudaFree(cD);
}

TEST(tensor2_matmul, Valid3x2) {
  cublasHandle_t handle;
  cublasCreate_v2(&handle);

  tensor2_t aT, bT, cT;
  tensor2_set_sizes(&aT, 3, 2);
  tensor2_set_sizes(&bT, 2, 3);
  tensor2_set_sizes(&cT, 3, 3);

  float aDh[3 * 2] = {
    1, 2, 
    3, 4,
    5, 6
  };

  float bDh[2 * 3] = {
    2, 3, 4,
    5, 6, 7
  };

  float cDh[3 * 3];
  float cDh_true[3 * 3] = {
    12, 15, 18,
    26, 33, 40,
    40, 51, 62
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

  AllFloatEqual(cDh, cDh_true, 3 * 3);

  cublasDestroy_v2(handle);
  cudaFree(aD);
  cudaFree(bD);
  cudaFree(cD);
}

TEST(tensor2_matmul, ValidTranspose3x2) {
  cublasHandle_t handle;
  cublasCreate_v2(&handle);

  tensor2_t aT, bT, cT;
  tensor2_set_sizes(&aT, 3, 2);
  tensor2_set_sizes(&bT, 2, 3);
  tensor2_set_sizes(&cT, 3, 3);

  float aDh[3 * 2] = {
    1, 2, 
    3, 4,
    5, 6
  };

  float bDh[2 * 3] = {
    2, 3, 4,
    5, 6, 7
  };

  float cDh[3 * 3];
  float cDh_true[3 * 3] = {
    12, 26, 40,
    15, 33, 51,
    18, 40, 62
  };

  float *aD, *bD, *cD;

  cudaMalloc(&aD, sizeof(float) * tensor2_get_size(&aT));
  cudaMalloc(&bD, sizeof(float) * tensor2_get_size(&bT));
  cudaMalloc(&cD, sizeof(float) * tensor2_get_size(&cT));

  cudaMemcpy(aD, aDh, sizeof(float) * tensor2_get_size(&aT), cudaMemcpyHostToDevice);
  cudaMemcpy(bD, bDh, sizeof(float) * tensor2_get_size(&bT), cudaMemcpyHostToDevice);

  tensor2_matmul(handle, CUBLAS_OP_T, CUBLAS_OP_T,
    bD, &bT, aD, &aT, cD, &cT);

  cudaMemcpy(cDh, cD, sizeof(float) * tensor2_get_size(&cT), cudaMemcpyDeviceToHost);

  AllFloatEqual(cDh, cDh_true, 3 * 3);

  cublasDestroy_v2(handle);
  cudaFree(aD);
  cudaFree(bD);
  cudaFree(cD);
}

TEST(tensor2_matmul, Valid3) {
  cublasHandle_t handle;
  cublasCreate_v2(&handle);

  tensor2_t aT, bT, cT;
  tensor2_set_sizes(&aT, 3, 2);
  tensor2_set_sizes(&bT, 2, 3);
  tensor2_set_sizes(&cT, 2, 2);

  float aDh[3 * 2] = {
    1, 2, 
    3, 4,
    5, 6
  };

  float bDh[2 * 3] = {
    2, 3, 4,
    5, 6, 7
  };

  float cDh[2 * 2];
  float cDh_true[2 * 2] = {
    31, 40,
    58, 76
  };

  float *aD, *bD, *cD;

  cudaMalloc(&aD, sizeof(float) * tensor2_get_size(&aT));
  cudaMalloc(&bD, sizeof(float) * tensor2_get_size(&bT));
  cudaMalloc(&cD, sizeof(float) * tensor2_get_size(&cT));

  cudaMemcpy(aD, aDh, sizeof(float) * tensor2_get_size(&aT), cudaMemcpyHostToDevice);
  cudaMemcpy(bD, bDh, sizeof(float) * tensor2_get_size(&bT), cudaMemcpyHostToDevice);

  tensor2_matmul(handle, CUBLAS_OP_N, CUBLAS_OP_N,
    bD, &bT, aD, &aT, cD, &cT);

  cudaMemcpy(cDh, cD, sizeof(float) * tensor2_get_size(&cT), cudaMemcpyDeviceToHost);

  AllFloatEqual(cDh, cDh_true, 2 * 2);

  cublasDestroy_v2(handle);
  cudaFree(aD);
  cudaFree(bD);
  cudaFree(cD);
}

int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
