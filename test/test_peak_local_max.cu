#include "gtest/gtest.h"

#include "test_utils.h"

#include "../src/peak_local_max.h"

TEST(peak_local_max, ComputesCorrectPeaks0) {
  const int cmap_channels = 2;
  const int cmap_width = 4;
  const int cmap_height = 4;
  const int peak_max_count = 5;
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
  
  int peak_counts[cmap_channels];
  int peak_indices[cmap_channels * peak_max_count];
  int *peak_ptrs[cmap_channels];
  for (int i = 0; i < cmap_channels; i++) {
    peak_ptrs[i] = &peak_indices[i * peak_max_count];
  }

  peak_local_max(cmap, cmap_channels, cmap_height, cmap_width,
      threshold, peak_counts, peak_ptrs, peak_max_count);

  int peak_true[peak_max_count] = { 5, 11, 13 };
  for (int i = 0; i < cmap_channels; i++) {
    ASSERT_EQ(3, peak_counts[i]);
    assert_all_equal(peak_ptrs[i], peak_true, peak_counts[i]);
  }
}

#ifndef EXCLUDE_MAIN
int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif
