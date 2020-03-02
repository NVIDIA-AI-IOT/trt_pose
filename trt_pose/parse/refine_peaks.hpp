#pragma once

namespace trt_pose {
namespace parse {

void refine_peaks_out_hw(float *refined_peaks, // Mx2
                         const int *counts,    // 1
                         const int *peaks,     // Mx2
                         const float *cmap,    // HxW
                         const int H, const int W, const int M,
                         const int window_size);

void refine_peaks_out_chw(float *refined_peaks, // CxMx2
                          const int *counts,    // C
                          const int *peaks,     // CxMx2
                          const float *cmap, const int C, const int H,
                          const int W, const int M, const int window_size);

void refine_peaks_out_nchw(float *refined_peaks, // NxCxMx2
                           const int *counts,    // NxC
                           const int *peaks,     // NxCxMx2
                           const float *cmap, const int N, const int C,
                           const int H, const int W, const int M,
                           const int window_size);

} // namespace parse
} // namespace trt_pose
