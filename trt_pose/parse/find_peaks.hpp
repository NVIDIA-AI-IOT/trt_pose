#pragma once

namespace trt_pose {
namespace parse {

void find_peaks_out_hw(int *counts,        // 1
                       int *peaks,         // Mx2
                       const float *input, // HxW
                       const int H, const int W, const int M,
                       const float threshold, const int window_size);

void find_peaks_out_chw(int *counts,        // C
                        int *peaks,         // CxMx2
                        const float *input, // CxHxW
                        const int C, const int H, const int W, const int M,
                        const float threshold, const int window_size);

void find_peaks_out_nchw(int *counts,        // NxC
                         int *peaks,         // NxCxMx2
                         const float *input, // NxCxHxW
                         const int N, const int C, const int H, const int W,
                         const int M, const float threshold,
                         const int window_size);

} // namespace parse
} // namespace trt_pose
