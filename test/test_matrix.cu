#include "gtest/gtest.h"

#include "../src/matrix.h"

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

int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
