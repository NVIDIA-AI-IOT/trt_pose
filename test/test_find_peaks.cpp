#include "gtest/gtest.h"

#include "../src/Matrix.hpp"
#include "../src/FindPeaks.hpp"

TEST(find_peaks, works)
{
  const int n = 4;
  const int m = 4;

  float data[n * m] = {
    0, 1, 0, 0,
    1, 2, 0, 0,
    0, 1, 0, 2,
    0, 0, 0, 0, 
  };

  Matrix<float> mat(data, n, m);
  auto peaks = findPeaks(mat, 1.0f);

  std::pair<int, int> p0 = {1, 1};
  std::pair<int, int> p1 = {2, 3};
  ASSERT_EQ(p0, peaks[0]);
  ASSERT_EQ(p1, peaks[1]);
}

#ifndef EXCLUDE_MAIN
int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif
