#pragma once

#include "matrix.h"

template<typename T>
int matrix_peak_threshold_mask_d(matrix_t *m, T *data, uint8_t *mask, T threshold, cudaStream_t streamId=NULL);

template<typename T>
int matrix_count_nonzero_h(matrix_t *m, T *data)
{
  int count = 0;
  T *max = data + matrix_size(m);
  while (data != max) 
  {
    if (*data != 0) {
      count++;
    }
    *data++;
  }
  return count;
}

template<typename T>
int matrix_index_nonzero_h(matrix_t *m, T *data, uint64_t *index)
{
  for (uint64_t i = 0; i < matrix_size(m); i++) {
    if (*data != 0) {
      *index = i;
      index++;
    }
    data++;
  }
  return 0;
}
