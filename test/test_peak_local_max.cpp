#include "gtest/gtest.h"
#include <iostream>

#include "test_utils.h"

#include "../src/peak_local_max.h"

TEST(peak_local_max, ComputesCorrectPeaks0) {
  
  float data[] = {
    0, 1, 0, 0,
    0, 2, 0, 0,
    0, 0, 0, 2,
    1, 2, 1, 0,
  };
  
  matrix_t m = { data, 4, 4 };
  const int max_num_peaks = 10;
  peak_t peaks[max_num_peaks];

  int num_peaks = peak_local_max(&m, 0.5, peaks, max_num_peaks);
  ASSERT_EQ(3, num_peaks);
  ASSERT_EQ(1, peaks[0].row);
  ASSERT_EQ(1, peaks[0].col);
  ASSERT_EQ(2, peaks[1].row);
  ASSERT_EQ(3, peaks[1].col);
  ASSERT_EQ(3, peaks[2].row);
  ASSERT_EQ(1, peaks[2].col);
}

#ifndef EXCLUDE_MAIN
int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif
