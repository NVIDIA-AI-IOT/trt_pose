#include "tensor.h"
#include "tensor_transpose.h"
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
    float *cD, tensor2_t *cT, float alpha=1.0f, float beta=0.0f)
{
  tensor2_t tmpT = tensor2_transpose(cT);
  float *tmp;
  cudaMalloc(&tmp, sizeof(float) * tensor2_get_size(&tmpT));

  tensor2_t aT_T = transa == CUBLAS_OP_N ? *aT : tensor2_transpose(aT);
  tensor2_t bT_T = transb == CUBLAS_OP_N ? *bT : tensor2_transpose(bT);

  // matmul C^T = AB
  cublasSgemm(
    handle,
    trans_inv(transa), // invert because cublas is column major
    trans_inv(transb),
    aT_T.sizes[0], bT_T.sizes[1], aT_T.sizes[1], 
    &alpha,
    aD, aT->sizes[1], // leading dim is stride based off column major indexing
    bD, bT->sizes[1],
    &beta,
    tmp, cT->sizes[1]
  );

  // transpose
  tensor2_transpose_data(handle, tmp, &tmpT, cD, cT);

  cudaFree(tmp);

  return 0;
}

int tensor3_matmul_batch(
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
