#include "peak_local_max.h"

#include "tensor.h"

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


