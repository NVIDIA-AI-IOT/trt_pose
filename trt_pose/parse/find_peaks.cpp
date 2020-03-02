#include "find_peaks.hpp"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

namespace trt_pose {
namespace parse {

void find_peaks_out_hw(int *counts,        // 1
                       int *peaks,         // Mx2
                       const float *input, // HxW
                       const int H, const int W, const int M,
                       const float threshold, const int window_size) {
  int win = window_size / 2;
  int count = 0;

  for (int i = 0; i < H && count < M; i++) {
    for (int j = 0; j < W && count < M; j++) {
      float val = input[i * W + j];

      // skip if below threshold
      if (val < threshold)
        continue;

      // compute window bounds
      int ii_min = MAX(i - win, 0);
      int jj_min = MAX(j - win, 0);
      int ii_max = MIN(i + win + 1, H);
      int jj_max = MIN(j + win + 1, W);

      // search for larger value in window
      bool is_peak = true;
      for (int ii = ii_min; ii < ii_max; ii++) {
        for (int jj = jj_min; jj < jj_max; jj++) {
          if (input[ii * W + jj] > val) {
            is_peak = false;
          }
        }
      }

      // add peak
      if (is_peak) {
        peaks[count * 2] = i;
        peaks[count * 2 + 1] = j;
        count++;
      }
    }
  }

  *counts = count;
}

void find_peaks_out_chw(int *counts,        // C
                        int *peaks,         // CxMx2
                        const float *input, // CxHxW
                        const int C, const int H, const int W, const int M,
                        const float threshold, const int window_size) {
  for (int c = 0; c < C; c++) {
    int *counts_c = &counts[c];
    int *peaks_c = &peaks[c * M * 2];
    const float *input_c = &input[c * H * W];
    find_peaks_out_hw(counts_c, peaks_c, input_c, H, W, M, threshold,
                      window_size);
  }
}

void find_peaks_out_nchw(int *counts,        // C
                         int *peaks,         // CxMx2
                         const float *input, // CxHxW
                         const int N, const int C, const int H, const int W,
                         const int M, const float threshold,
                         const int window_size) {
  for (int n = 0; n < N; n++) {
    int *counts_n = &counts[n * C];
    int *peaks_n = &peaks[n * C * M * 2];
    const float *input_n = &input[n * C * H * W];
    find_peaks_out_chw(counts_n, peaks_n, input_n, C, H, W, M, threshold,
                       window_size);
  }
}

} // namespace parse
} // namespace trt_pose
