#include "tensor.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"

inline cublasOperation_t trans_inv(cublasOperation_t trans)
{
  return trans == CUBLAS_OP_N ? CUBLAS_OP_T : CUBLAS_OP_N;
}

int tensor2_matmul(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    float *aD, tensor2_t *aT, 
    float *bD, tensor2_t *bT,
    float *cD, float *cW, tensor2_t *cT, float alpha=1.0f, float beta=0.0f)
{
  // allocate workspace for tranpose
  cublasSgemm(
    handle,
    trans_inv(transa), // invert because cublas is column major
    trans_inv(transb),
    aT->sizes[0], bT->sizes[1], aT->sizes[1], 
    &alpha,
    aD, aT->strides[0], // leading dim is stride based off column major indexing
    bD, bT->strides[0],
    &beta,
    cW, cT->strides[0]
  );

  return 0;
};

int tensor2_matmul(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    float *aD, tensor2_t *aT, 
    float *bD, tensor2_t *bT,
    float *cD, tensor2_t *cT, float alpha=1.0f, float beta=0.0f)
{
  float *cW;
  cudaMalloc(&cW, sizeof(float) * tensor2_get_size(cT));
  tensor2_matmul(handle, transa, transb, aD, aT, bD, bT, cD, cW, cT, alpha, beta);
  cudaFree(cW);

  return 0;
}

int tensor3_matmul_batch_cuda(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    float *aD, tensor3_t *aT,
    float *bD, tensor3_t *bT,
    float *cD, tensor3_t *cT,
    float alpha=1.0f, float beta=0.0f)
{

  // generate pointer list
  // copy ptr list to device
  // perform batch matrix multiplication
  return 0;
}
