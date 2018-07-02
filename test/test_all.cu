#define EXCLUDE_MAIN

#include "test_matrix.cu"
#include "test_matrix_fill.cu"
#include "test_matrix_multiply.cu"
#include "test_matrix_peak_threshold.cu"
#include "test_matrix_solve.cu"

int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
