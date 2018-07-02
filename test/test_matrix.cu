#include "gtest/gtest.h"

#include "../src/matrix.h"
#include "../src/matrix_index.cuh"
#include "../src/matrix_copy.h"

TEST(matrix_index, Valid)
{
  matrix_t m;
  matrix_set_shape(&m, 3, 5);

  // test column major indexing
  ASSERT_EQ(0, matrix_index_c(&m, 0, 0)); 
  ASSERT_EQ(1, matrix_index_c(&m, 1, 0)); 
  ASSERT_EQ(3, matrix_index_c(&m, 0, 1)); 
  ASSERT_EQ(4, matrix_index_c(&m, 1, 1)); 
  ASSERT_EQ(6, matrix_index_c(&m, 0, 2)); 

  // test row major indexing
  ASSERT_EQ(0, matrix_index_r(&m, 0, 0)); 
  ASSERT_EQ(5, matrix_index_r(&m, 1, 0)); 
  ASSERT_EQ(1, matrix_index_r(&m, 0, 1)); 
  ASSERT_EQ(6, matrix_index_r(&m, 1, 1)); 
  ASSERT_EQ(2, matrix_index_r(&m, 0, 2)); 
}

TEST(matrix_size, Valid)
{
  matrix_t m;
  matrix_set_shape(&m, 3, 5);
  ASSERT_EQ(15, matrix_size(&m));
}

TEST(matrix_copy_h2h_transpose, Valid)
{
  matrix_t m;
  matrix_set_shape(&m, 3, 2);
  float dT[] = {
    0, 1,
    2, 3,
    4, 5
  };
  float d[3 * 2];
  matrix_copy_h2h_transpose(&m, dT, d);
  ASSERT_EQ(0, d[0]);
  ASSERT_EQ(2, d[1]);
  ASSERT_EQ(4, d[2]);
  ASSERT_EQ(1, d[3]);
  ASSERT_EQ(0, d[matrix_index_c(&m, 0, 0)]);
  ASSERT_EQ(2, d[matrix_index_c(&m, 1, 0)]);
  ASSERT_EQ(5, d[matrix_index_c(&m, 2, 1)]);
}

#ifndef EXCLUDE_MAIN
int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif
