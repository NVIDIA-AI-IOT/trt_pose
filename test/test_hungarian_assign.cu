#include "gtest/gtest.h"
#include "test_utils.h"

#include "../src/hungarian_assign.h"

TEST(count_zeros_in_row, Valid) {
  const int N = 3;
  const int M = 4;
  int C[3 * 4] = {
    0, 1, 2, 0,
    -1, 0, 1, 0,
    0, 0, 0, 3
  };
  ASSERT_EQ(2, count_zeros_in_row(C, N, M, 0));
  ASSERT_EQ(2, count_zeros_in_row(C, N, M, 1));
  ASSERT_EQ(3, count_zeros_in_row(C, N, M, 2));
}

TEST(count_zeros_in_col, Valid) {
  const int N = 3;
  const int M = 4;
  int C[3 * 4] = {
    0, 1, 2, 0,
    -1, 0, 1, 0,
    0, 0, 0, 3
  };
  ASSERT_EQ(2, count_zeros_in_col(C, N, M, 0));
  ASSERT_EQ(2, count_zeros_in_col(C, N, M, 1));
  ASSERT_EQ(1, count_zeros_in_col(C, N, M, 2));
  ASSERT_EQ(2, count_zeros_in_col(C, N, M, 3));
}

TEST(count_zeros, Valid) {
  const int N = 3;
  const int M = 4;
  int C[3 * 4] = {
    0, 1, 2, 0,
    -1, 0, 1, 0,
    0, 0, 0, 3
  };
  ASSERT_EQ(7, count_zeros(C, N, M));
}

TEST(hungarian_first_zero_in_row, Valid) {
  const int N = 3;
  const int M = 4;
  int C[3 * 4] = {
    0, 1, 2, 0,
    -1, 0, 1, 0,
    2, 3, 4, 3
  };
  ASSERT_EQ(0, hungarian_first_zero_in_row(C, N, M, 0));
  ASSERT_EQ(1, hungarian_first_zero_in_row(C, N, M, 1));
  ASSERT_EQ(-1, hungarian_first_zero_in_row(C, N, M, 2));
}

TEST(hungarian_first_zero_in_col, Valid) {
  const int N = 3;
  const int M = 4;
  int C[3 * 4] = {
    0, 1, 2, 0,
    -1, 0, 1, 0,
    2, 3, 4, 3
  };
  ASSERT_EQ(0, hungarian_first_zero_in_col(C, N, M, 0));
  ASSERT_EQ(1, hungarian_first_zero_in_col(C, N, M, 1));
  ASSERT_EQ(-1, hungarian_first_zero_in_col(C, N, M, 2));
  ASSERT_EQ(0, hungarian_first_zero_in_col(C, N, M, 3));
}

TEST(hungarian_cross_out_row, Valid) {
  const int N = 3;
  const int M = 4;
  int C[3 * 4] = {
    0, 1, 2, 0,
    -1, 0, 1, 0,
    2, 3, 4, 3
  };
  int C_cross_true[3 * 4] = {
    -1, -1, -1, -1,
    -1, 0, 1, 0,
    2, 3, 4, 3
  };
  hungarian_cross_out_row(C, N, M, 0);
  assert_all_equal(C_cross_true, C, 3 * 4);
}

TEST(hungarian_cross_out_col, Valid) {
  const int N = 3;
  const int M = 4;
  int C[3 * 4] = {
    0, 1, 2, 0,
    -1, 0, 1, 0,
    2, 3, 4, 3
  };
  int C_cross_true[3 * 4] = {
    0, 1, -1, 0,
    -1, 0, -1, 0,
    2, 3, -1, 3
  };
  hungarian_cross_out_col(C, N, M, 2);
  assert_all_equal(C_cross_true, C, 3 * 4);
}

TEST(hungarian_assign_single_zero_row, ShouldNotAssign0) 
{
  const int N = 3;
  const int M = 4;
  int C[3 * 4] = {
    0, 1, 2, 0,
    1, 0, 1, 0,
    2, 3, 4, 3
  };
  ASSERT_EQ(false, hungarian_assign_single_zero_row(C, N, M));
}

TEST(hungarian_assign_single_zero_row, ShouldNotAssign1) 
{
  const int N = 3;
  const int M = 4;
  int C[3 * 4] = {
    0, 1, 2, 0,
    1, 0, 1, 0,
    0, 3, 0, 3
  };
  ASSERT_EQ(false, hungarian_assign_single_zero_row(C, N, M));
}

TEST(hungarian_assign_single_zero_row, ShouldAssign0) 
{
  const int N = 3;
  const int M = 4;
  int C[3 * 4] = {
    0, 1, 2, 0,
    1, 0, 1, 0,
    0, 3, -1, 3
  };
  int C_after[3*4] = {
    -1, 1, 2, 0,
    -1, 0, 1, 0,
     -2, -1, -1, -1,
  };
  ASSERT_EQ(true, hungarian_assign_single_zero_row(C, N, M));
  assert_all_equal(C_after, C, 3 * 4);
}

TEST(hungarian_assign_single_zero_col, ShouldAssign0) 
{
  const int N = 3;
  const int M = 4;
  int C[3 * 4] = {
    0, 1, 2, 0,
    1, 0, 1, 0,
    0, 3, 1, 3
  };
  int C_after[3*4] = {
    0, -1, 2, 0,
    -1, -2, -1, -1,
    0, -1, 1, 3,
  };
  ASSERT_EQ(true, hungarian_assign_single_zero_col(C, N, M));
  assert_all_equal(C_after, C, 3 * 4);
}

TEST(hungarian_assign, Valid_v0) {
  const int N = 3;
  const int M = 4;
  int C[3 * 4] = {
    0, 1, 2, 0,
    1, 0, 1, 0,
    0, 3, 1, 3
  };
  int C_after[3*4] = {
    -1, -1, -1, -2,
    -1, -2, -1, -1,
    -2, -1, -1, -1,
  };
  hungarian_assign(C, N, M);
  assert_all_equal(C_after, C, 3 * 4);
}

TEST(hungarian_assign, Valid_v1) {
  const int N = 3;
  const int M = 4;
  int C[3 * 4] = {
    0, 1, 0, 0,
    1, 0, 0, 0,
    0, 3, 1, 3
  };
  int C_after[3*4] = {
    -1, -1, -2, -1,
    -1, -2, -1, -1,
    -2, -1, -1, -1,
  };
  hungarian_assign(C, N, M);
  assert_all_equal(C_after, C, 3 * 4);
}

TEST(hungarian_assign, Valid_v2) {
  const int N = 3;
  const int M = 4;
  int C[3 * 4] = {
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0
  };
  int C_after[3*4] = {
    -2, -1, -1, -1,
    -1, -2, -1, -1,
    -1, -1, -2, -1,
  };
  hungarian_assign(C, N, M);
  assert_all_equal(C_after, C, 3 * 4);
}

TEST(hungarian_assign, Valid_v3) {
  const int N = 3;
  const int M = 4;
  int C[3 * 4] = {
    0, 0, 0, 0,
    0, 0, 2, 3,
    0, 0, 1, 4
  };
  int C_after[3*4] = {
    -1, -1, -2, -1,
    -2, -1, -1, -1,
    -1, -2, -1, -1,
  };
  hungarian_assign(C, N, M);
  assert_all_equal(C_after, C, 3 * 4);
}

TEST(hungarian_check_optimality, IsOptimal_v0) {
  const int N = 3;
  const int M = 4;
  int C[3 * 4] = {
    0, 0, 0, 0,
    0, 0, 2, 3,
    0, 0, 1, 4
  };
  ASSERT_EQ(true, hungarian_check_optimality(C, N, M));
};

TEST(hungarian_check_optimality, IsOptimal_v1) {
  const int N = 3;
  const int M = 4;
  int C[3 * 4] = {
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0
  };
  ASSERT_EQ(true, hungarian_check_optimality(C, N, M));
};

TEST(hungarian_check_optimality, IsOptimal_v2) {
  const int N = 3;
  const int M = 4;
  int C[3 * 4] = {
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0
  };
  ASSERT_EQ(true, hungarian_check_optimality(C, N, M));
};

TEST(hungarian_check_optimality, NotOptimal_v0) {
  const int N = 3;
  const int M = 4;
  int C[3 * 4] = {
    0, 0, 0, 0,
    0, 1, 2, 1,
    0, 1, 1, 1
  };
  ASSERT_EQ(false, hungarian_check_optimality(C, N, M));
};

TEST(hungarian_check_optimality, NotOptimal_v1) {
  const int N = 3;
  const int M = 4;
  int C[3 * 4] = {
    0, 0, 1, 1,
    0, 0, 2, 1,
    0, 0, 1, 1
  };
  ASSERT_EQ(false, hungarian_check_optimality(C, N, M));
};

TEST(hungarian_generate_mask, Valid_v0) {
  const int N = 3;
  const int M = 4;
  float G[3 * 4] = {
    0.0, 0.1, 2.0, 0.3,
    0.2, 0.3, 0.0, 1.1,
    0.0, 0.0, 1.0, 2.0
  };
  int C[3 * 4] = {
    0, 1, 1, 1,
    1, 1, 0, 1,
    0, 0, 1, 1
  };
  hungarian_generate_mask(G, N, M, C);
}

TEST(hungarian_generate_mask, Valid_v1) {
  const int N = 3;
  const int M = 4;
  float G[3 * 4] = {
    0.0, -0.1, -2.0, -0.3,
    -0.2, -0.3, 0.0, -1.1,
    0.0, 0.0, -1.0, -2.0
  };
  int C[3 * 4] = {
    0, 1, 1, 1,
    1, 1, 0, 1,
    0, 0, 1, 1
  };
  hungarian_generate_mask(G, N, M, C);
}

TEST(hungarian_sub_min_row, Valid) {
  const int N = 3;
  const int M = 4;
  float G[3 * 4] = {
    1, 2, 3, 1,
    3, 4, 2, 5,
    4, 5, 1, 2
  };
  float G_true[3 * 4] = {
    0, 1, 2, 0,
    1, 2, 0, 3,
    3, 4, 0, 1
  };
  hungarian_sub_min_row(G, N, M);
  assert_all_equal(G_true, G, N * M);
}

TEST(hungarian_sub_min_col, Valid) {
  const int N = 3;
  const int M = 4;
  float G[3 * 4] = {
    1, 2, 3, 1,
    3, 4, 2, 5,
    4, 5, 1, 2
  };
  float G_true[3 * 4] = {
    0, 0, 2, 0,
    2, 2, 1, 4,
    3, 3, 0, 1
  };
  hungarian_sub_min_col(G, N, M);
  assert_all_equal(G_true, G, N * M);
}

TEST(hungarian_sub_min_row, ValidInverse) {
  const int N = 3;
  const int M = 4;
  float G[3 * 4] = {
    -1, -2, -3, -1,
    -3, -4, -2, -5,
    -4, -5, -1, -2
  };
  float G_true[3 * 4] = {
    2, 1, 0, 2,
    2, 1, 3, 0,
    1, 0, 4, 3
  };
  hungarian_sub_min_row(G, N, M);
  assert_all_equal(G_true, G, N * M);
}

TEST(hungarian_sub_min_col, ValidInverse) {
  const int N = 3;
  const int M = 4;
  float G[3 * 4] = {
    -1, -2, -3, -1,
    -3, -4, -2, -5,
    -4, -5, -1, -2
  };
  float G_true[3 * 4] = {
    3, 3, 0, 4,
    1, 1, 1, 0,
    0, 0, 2, 3
  };
  hungarian_sub_min_col(G, N, M);
  assert_all_equal(G_true, G, N * M);
}

#ifndef EXCLUDE_MAIN
int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif
