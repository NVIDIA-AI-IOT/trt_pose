#include "gtest/gtest.h"

#include <cstdint>

struct Array
{
  float *data;
  uint64_t size;
};

__global__ void DoubleKernel(Array array)
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  array.data[idx] *= 2.0f;
}

void Double(Array array) 
{
  DoubleKernel<<<1, array.size>>>(array);  
}

TEST(Double, DoubleDoubles)
{
  Array arrayHost;
  arrayHost.data = (float*) malloc(sizeof(float) * 5);
  arrayHost.size = 5;

  Array array;
  array.size = arrayHost.size;

  for (uint64_t i = 0; i < arrayHost.size; i++) {
    arrayHost.data[i] = i;
  }

  cudaMalloc(&array.data, sizeof(float) * array.size);
  cudaMemcpy(array.data, arrayHost.data, sizeof(float) * arrayHost.size, cudaMemcpyHostToDevice);

  Double(array);

  cudaMemcpy(arrayHost.data, array.data, sizeof(float) * arrayHost.size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < arrayHost.size; i++) {
    ASSERT_FLOAT_EQ(i * 2.0f, arrayHost.data[i]);
  }

  free(arrayHost.data);
  cudaFree(array.data);
}

int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
