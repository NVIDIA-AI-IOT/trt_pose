#pragma once

#include "cublas_v2.h"
#include "tensor.h"

inline tensor2_t tensor2_transpose(tensor2 *t)
{
  tensor2_t tmp;
  tmp.sizes[0] = t->sizes[1];
  tmp.sizes[1] = t->sizes[0];
  tensor2_set_linear_strides(&tmp);
  return tmp;
}

int tensor2_transpose_data(cublasHandle_t handle, 
    float *aD, tensor2_t *aT, float *bD, tensor2_t *bT)
{
  *bT = tensor2_transpose(aT);

  // transpose c
  float alpha_t = 1.0f;
  float beta_t = 0.0f;

  cublasSgeam(
    handle,
    CUBLAS_OP_T, CUBLAS_OP_T,
    aT->sizes[0], aT->sizes[1],
    &alpha_t,
    aD, aT->strides[0],
    &beta_t,
    aD, aT->strides[0],
    bD, bT->strides[0]
  );

  return 0;
}
