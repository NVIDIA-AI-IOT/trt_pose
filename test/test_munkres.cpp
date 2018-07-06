#include "gtest/gtest.h"

#include "test_utils.h"

#include "../src/munkres.h"
#include "../src/tensor.h"

TEST(munkres_sub_min_row, Correct)
{
  const int n = 3;
  const int m = 4;
  float a[n * m] = {
    1, 2, 3, 2,
    2, 1, 3, 1,
    3, 4, 2, 3
  };
  float a_1[n * m] = {
    0, 1, 2, 1,
    1, 0, 2, 0,
    1, 2, 0, 1,
  };

  _munkres_sub_min_row(a, n, m);
  assert_all_equal(a_1, a, n * m);
}

TEST(munkres_sub_min_col, Correct)
{
  const int n = 3;
  const int m = 4;
  float a[n * m] = {
    1, 2, 3, 2,
    2, 1, 3, 1,
    3, 4, 2, 3
  };
  float a_1[n * m] = {
    0, 1, 1, 1,
    1, 0, 1, 0,
    2, 3, 0, 2,
  };

  _munkres_sub_min_col(a, n, m);
  assert_all_equal(a_1, a, n * m);
}

TEST(munkres_step_1, Correct)
{
  const int n = 3;
  const int m = 4;

  float a[n * m] = {
    0, 1, 1, 1,
    1, 0, 1, 0,
    2, 3, 0, 2,
  };

  int s[n * m] = {
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
  };

  int s_1[n * m] = {
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
  };

  _munkres_step_1(a, s, n, m);
  assert_all_equal(s_1, s, n * m);
}

TEST(munkres_step_2, ShouldReturnTrue)
{
  const int n = 3;
  const int m = 4;

  int s[n * m] = {
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
  };
  int c1[m] = { 
    0, 0, 0, 0
  };

  int done = _munkres_step_2(s, c1, n, m);
  ASSERT_EQ(1, done);
}

TEST(munkres_step_2, ShouldReturnFalse)
{
  const int n = 3;
  const int m = 4;

  int s[n * m] = {
    1, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 1, 0,
  };
  int c1[m] = { 
    0, 0, 0, 0
  };

  int done = _munkres_step_2(s, c1, n, m);
  ASSERT_EQ(0, done);
}

TEST(munkres_step_3_all_covered, Correct)
{
  const int n = 3;
  const int m = 4;

  float a[n * m] = {
    0, 1, 3, 2,
    1, 0, 2, 0,
    1, 3, 0, 1
  };

  {
  int c0[n] = { 0, 0, 0 };
  int c1[m] = { 0, 0, 0, 0 };
  ASSERT_EQ(0, _munkres_step_3_all_covered(a, c0, c1, n, m));
  }

  {
  int c0[n] = { 1, 1, 1 };
  int c1[m] = { 0, 0, 0, 0 };
  ASSERT_EQ(1, _munkres_step_3_all_covered(a, c0, c1, n, m));
  }

  {
  int c0[n] = { 0, 0, 0 };
  int c1[m] = { 1, 1, 1, 1 };
  ASSERT_EQ(1, _munkres_step_3_all_covered(a, c0, c1, n, m));
  }

  {
  int c0[n] = { 0, 0, 0 };
  int c1[m] = { 1, 0, 1, 1 };
  ASSERT_EQ(0, _munkres_step_3_all_covered(a, c0, c1, n, m));
  }

  {
  int c0[n] = { 1, 0, 1 };
  int c1[m] = { 0, 1, 0, 1 };
  ASSERT_EQ(1, _munkres_step_3_all_covered(a, c0, c1, n, m));
  }
}

TEST(munkres_step_3_prime, Correct)
{
  const int n = 3;
  const int m = 4;

  float a[n * m] = {
    0, 1, 0, 2,
    1, 0, 2, 3,
    1, 0, 1, 1
  };

  int s[n * m] = {
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 0, 0
  };

  int c0[n] = {
    0, 0, 0 
  };
  int c1[m] = { 
    1, 1, 0, 0
  };

  int p[n * m] = {
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
  };
  int p0, p1;
  int step4 = _munkres_step_3_prime(a, c0, c1, s, p, &p0, &p1, n, m);
  ASSERT_EQ(0, step4);
}

TEST(munkres_step_3_prime, Correct2)
{
  const int n = 3;
  const int m = 4;

  float a[n * m] = {
    0, 0, 1, 0,
    0, 0, 0, 1,
    0, 1, 0, 0
  };

  int s[n * m] = {
    1, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0
  };

  int c0[n] = {
    1, 0, 0 
  };
  int c1[m] = { 
    0, 0, 0, 0
  };

  int p[n * m] = {
    0, 1, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
  };

  int p0, p1;
  int step4 = _munkres_step_3_prime(a, c0, c1, s, p, &p0, &p1, n, m);
  ASSERT_EQ(1, step4);
  ASSERT_EQ(1, p[IDX_2D(1, 0, m)]);
}

TEST(munkres_step_3, Correct)
{
  const int n = 3;
  const int m = 4;

  float a[n * m] = {
    0, 0, 1, 0,
    0, 0, 0, 1,
    0, 1, 0, 0
  };

  int s[n * m] = {
    1, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0
  };

  int c0[n] = {
    0, 0, 0 
  };
  int c1[m] = { 
    1, 0, 0, 0
  };

  int p[n * m] = {
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
  };

  int p0, p1;
  int step4 = _munkres_step_3(a, c0, c1, s, p, &p0, &p1, n, m);
  ASSERT_EQ(1, step4);
  ASSERT_EQ(1, p[IDX_2D(1, 0, m)]);
  ASSERT_EQ(1, p[IDX_2D(0, 1, m)]);
}

TEST(munkres_step_4, Correct)
{
  const int n = 3;
  const int m = 4;

  float a[n * m] = {
    0, 0, 1, 0,
    0, 0, 0, 1,
    0, 1, 0, 0
  };

  int s[n * m] = {
    1, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0
  };

  int c0[n] = {
    1, 0, 0 
  };
  int c1[m] = { 
    0, 0, 0, 0
  };

  int p[n * m] = {
    0, 1, 0, 0,
    1, 0, 0, 0,
    0, 0, 0, 0,
  };

  int s_1[n * m] = {
    0, 1, 0, 0,
    1, 0, 0, 0,
    0, 0, 0, 0
  };

  int p0 = 1;
  int p1 = 0;
  _munkres_step_4(c0, c1, s, p, &p0, &p1, n, m);

  assert_all_equal(s_1, s, n * m);
  ASSERT_EQ(0, p0);
  ASSERT_EQ(1, p1);
}

TEST(munkres_step_5, Correct)
{
  const int n = 3;
  const int m = 4;

  float a[n * m] = {
    0, 2, 3, 2,
    0, 2, 2, 3,
    0, 3, 2, 2
  };

  int c0[n] = {
    0, 0, 0 
  };

  int c1[m] = { 
    1, 0, 0, 0
  };

  float a_1[n * m] = {
    0, 0, 1, 0,
    0, 0, 0, 1,
    0, 1, 0, 0
  };

  _munkres_step_5(a, c0, c1, n, m);
  assert_all_equal(a_1, a, n * m);
}

TEST(munkres, Correct)
{
  const int n = 3;
  const int m = 4;

  float a[n * m] = {
    1, 2, 3, 2,
    1, 2, 2, 3,
    1, 3, 2, 2
  };

  int c0[n] = {
    0, 0, 0 
  };

  int c1[m] = { 
    0, 0, 0, 0
  };

  int s[n * m] = {
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0
  };

  int p[n * m] = {
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0
  };

  int s_1[n * m] = {
    0, 0, 0, 1,
    0, 1, 0, 0,
    1, 0, 0, 0
  };

  _munkres(a, c0, c1, s, p, n, m);
  assert_all_equal(s_1, s, n * m);
}

TEST(munkres, CorrectWrapper)
{
  const int n = 3;
  const int m = 4;

  float a[n * m] = {
    1, 2, 3, 2,
    1, 2, 2, 3,
    1, 3, 2, 2
  };

  int s[n * m] = {
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0
  };

  int s_1[n * m] = {
    0, 0, 0, 1,
    0, 1, 0, 0,
    1, 0, 0, 0
  };

  size_t workspace_size = munkres_workspace_size(n, m);
  void *workspace = malloc(workspace_size);
  munkres(a, s, n, m, workspace, workspace_size);
  assert_all_equal(s_1, s, n * m);
}

#ifndef EXCLUDE_MAIN
int main(int argc, char *argv[]) 
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif
