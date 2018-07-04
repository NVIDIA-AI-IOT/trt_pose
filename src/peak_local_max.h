#pragma once

#include "tensor.h"

/**
 * Compute the location of local maxima in cmap that surpass threshold.
 * Returns the scalar index relative to the cmap channel.
 *
 * cmap (host | in) - The confidence map of size (CxHxW)
 * cmap_channels (in) - Number of channels in cmap
 * cmap_height (in) - Height of cmap
 * cmap_width (in) - Width of cmap
 * threshold (in) - The threshold peaks must surpass.
 * peak_counts (host | out) - Pointer to an integer array of length cmap_channels, that contains the number of peaks per each channel.
 * peak_ptrs (host | in/out) - Pointer to an array of integer pointers which are lists of peak indexes for each channel.  
 *   The memory to store peaks should be allocated to fit at least peak_max_count peaks per channel.
 * peak_max_count (host | in) - The maximum number of peaks per channel to detect.
 */
void peak_local_max(
    float *cmap, int cmap_channels, int cmap_height, int cmap_width,
    float threshold,
    int *peak_counts, int **peak_ptrs, int peak_max_count)
{
  for (int i = 0; i < cmap_channels; i++) {
    float *cmap_i = cmap + i * cmap_height * cmap_width;
    peak_counts[i] = 0;
    for (int j = 0; j < cmap_height; j++) {
      for (int k = 0; k < cmap_width; k++) {
        int idx = IDX_2D(j, k, cmap_width);
        float val = cmap_i[idx];

        // below threshold
        if (val < threshold) {
          continue;
        }

        // greater neighbor
        if (j - 1 > 0 && cmap_i[IDX_2D(j - 1, k, cmap_width)] > val) {
          continue;
        }
        if (j + 1 > 0 && cmap_i[IDX_2D(j + 1 , k, cmap_width)] > val) {
          continue;
        }
        if (k - 1 > 0 && cmap_i[IDX_2D(j, k - 1, cmap_width)] > val) {
          continue;
        }
        if (k + 1 > 0 && cmap_i[IDX_2D(j, k + 1, cmap_width)] > val) {
          continue;
        }

        // is peak - increment count for channel and set index
        if (peak_counts[i] < peak_max_count) { 
          peak_ptrs[i][peak_counts[i]] = idx; 
          peak_counts[i]++;
        }
      }
    }
  }
}
