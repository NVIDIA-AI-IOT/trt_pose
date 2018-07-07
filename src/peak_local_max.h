#ifndef PEAK_LOCAL_MAX_H
#define PEAK_LOCAL_MAX_H

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

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
void peak_local_max(const float *cmap, int cmap_channels, int cmap_height, int cmap_width,
    float threshold, int *peak_counts, int **peak_ptrs, int peak_max_count);

#ifdef __cplusplus
}
#endif

#endif // PEAK_LOCAL_MAX_H
