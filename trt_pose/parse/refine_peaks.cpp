#include "refine_peaks.hpp"

namespace trt_pose {
namespace parse {

inline int reflect(int idx, int min, int max) {
  if (idx < min) {
    return -idx;
  } else if (idx >= max) {
    return max - (idx - max) - 2;
  } else {
    return idx;
  }
}

void refine_peaks_out_hw(float *refined_peaks, // Mx2
                         const int *counts,    // 1
                         const int *peaks,     // Mx2
                         const float *cmap,    // HxW
                         const int H, const int W, const int M,
                         const int window_size) {
  int count = *counts;
  int win = window_size / 2;

  for (int m = 0; m < count; m++) {
    float *refined_peak = &refined_peaks[m * 2];
    refined_peak[0] = 0.;
    refined_peak[1] = 0.;
    const int *peak = &peaks[m * 2];

    int i = peak[0];
    int j = peak[1];
    float weight_sum = 0.;

    for (int ii = i - win; ii < i + win + 1; ii++) {
      int ii_idx = reflect(ii, 0, H);
      for (int jj = j - win; jj < j + win + 1; jj++) {
        int jj_idx = reflect(jj, 0, W);

        float weight = cmap[ii_idx * W + jj_idx];
        refined_peak[0] += weight * ii;
        refined_peak[1] += weight * jj;
        weight_sum += weight;
      }
    }

    refined_peak[0] /= weight_sum;
    refined_peak[1] /= weight_sum;
    refined_peak[0] += 0.5; // center pixel
    refined_peak[1] += 0.5; // center pixel
    refined_peak[0] /= H;   // normalize coordinates
    refined_peak[1] /= W;   // normalize coordinates
  }
}

void refine_peaks_out_chw(float *refined_peaks, // CxMx2
                          const int *counts,    // C
                          const int *peaks,     // CxMx2
                          const float *cmap, const int C, const int H,
                          const int W, const int M, const int window_size) {
  for (int c = 0; c < C; c++) {
    refine_peaks_out_hw(&refined_peaks[c * M * 2], &counts[c],
                        &peaks[c * M * 2], &cmap[c * H * W], H, W, M,
                        window_size);
  }
}

void refine_peaks_out_nchw(float *refined_peaks, // NxCxMx2
                           const int *counts,    // NxC
                           const int *peaks,     // NxCxMx2
                           const float *cmap, const int N, const int C,
                           const int H, const int W, const int M,
                           const int window_size) {
  for (int n = 0; n < N; n++) {
    refine_peaks_out_chw(&refined_peaks[n * C * M * 2], &counts[n * C],
                         &peaks[n * C * M * 2], &cmap[n * C * H * W], C, H, W,
                         M, window_size);
  }
}

} // namespace parse
} // namespace trt_pose
