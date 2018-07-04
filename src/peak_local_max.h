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
    const float *cmap, int cmap_channels, int cmap_height, int cmap_width,
    float threshold,
    int *peak_counts, int **peak_ptrs, int peak_max_count)
{
  for (int c = 0; c < cmap_channels; c++)
  {
    peak_counts[c] = 0;
    int c_idx = c * cmap_height * cmap_width;
    for (int i = 0; i < cmap_height; i++) 
    {
      int i_idx = c_idx + i * cmap_width;
      for (int j = 0; j < cmap_width; j++)
      {
        int idx = i_idx + j;
        float val = cmap[idx];

        if (val < threshold) {
          continue;
        }

        if ((j - 1 >= 0 && cmap[idx - 1] > val) ||
            (j + 1 < cmap_width && cmap[idx + 1] > val) ||
            (i - 1 >= 0 && cmap[idx - cmap_width] > val) ||
            (i + 1 < cmap_height && cmap[idx + cmap_width] > val))
        {
          continue;
        }
        
        if (peak_counts[c] != peak_max_count) {
          peak_ptrs[c][peak_counts[c]++] = i * cmap_width + j;
        } else {
          return;
        }
      }
    }
  } 
}
