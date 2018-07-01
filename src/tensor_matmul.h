#include "tensor.h"
#include "cublas_v2.h"

inline cublasOperation_t trans_inv(cublasOperation_t trans)
{
  return trans == CUBLAS_OP_N ? CUBLAS_OP_T : CUBLAS_OP_N;
}

int tensor2_matmul_cuda(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    float *aD, tensor2_t *aT, 
    float *bD, tensor2_t *bT,
    float *cD, tensor2_t *cT, float alpha=1.0f, float beta=0.0f)
{
  cublasSgemm(
    handle,
    trans_inv(transa), // invert because cublas is column major
    trans_inv(transb),
    aT->sizes[0], bT->sizes[1], aT->sizes[1], 
    &alpha,
    aD, aT->strides[0], // leading dim is stride based off column major indexing
    bD, bT->strides[0],
    &beta,
    cD, cT->strides[0]
  );

  return 0;
};
