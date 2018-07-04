#include "gtest/gtest.h"

#include "test_utils.h"

#include "../src/peak_local_max.h"

TEST(peak_local_max, ComputesCorrectPeaks0) {
  const int cmap_channels = 2;
  const int cmap_width = 4;
  const int cmap_height = 4;
  const int peak_max_count = 10;
  const float threshold = 0.5;
  
  float cmap[] = {
    0, 1, 0, 0,
    0, 2, 0, 0,
    0, 0, 0, 2,
    1, 2, 1, 0,
    0, 1, 0, 0,
    0, 2, 0, 0,
    0, 0, 0, 2,
    1, 2, 1, 0
  };
  
  int peak_indices[peak_max_count];

  int peak_count = peak_local_max(cmap, cmap_channels, cmap_height, cmap_width,
      threshold, peak_indices, peak_max_count);

  int peak_indices_true[peak_max_count] = { 5, 11, 13, 5 + 16, 11 + 16, 13 + 16 };

  ASSERT_EQ(6, peak_count);
  assert_all_equal(peak_indices, peak_indices_true, peak_count);
}

#ifndef EXCLUDE_MAIN
int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif
