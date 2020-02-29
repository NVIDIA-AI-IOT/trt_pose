#include "find_peaks.hpp"
#include <stdexcept>


void test_find_peaks_out_hw()
{
  const int N = 1;
  const int C = 2;
  const int H = 4;
  const int W = 4;
  const int M = 10;
  const float threshold = 2.0;
  const int window_size = 3;

  int counts[N * C];
  int peaks[N * C * M * 2];
  float input[N * C * H * W] = {

    0., 0., 0., 0.,
    0., 0., 3., 0.,
    0., 0., 0., 0.,
    1., 0., 0., 0.,

    0., 0., 0., 0.,
    0., 0., 0., 0.,
    0., 0., 0., 0.,
    0., 0., 0., 0.

  };

  find_peaks_out_nchw(counts, peaks, input, N, C, H, W, M, threshold, window_size);

  if (counts[0] != 1) {
    throw std::runtime_error("Number of peaks should be 1.");
  }
  if (peaks[0] != 1) {
    throw std::runtime_error("Peak i coordinate should be 1.");
  }
  if (peaks[1] != 2) {
    throw std::runtime_error("Peak j coordinate should be 2,");
  }
}

int main()
{
  test_find_peaks_out_hw();
  return 0;
}
