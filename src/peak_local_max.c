#include "peak_local_max.h"

#include "matrix.h"
#include "tensor.h"

int peak_local_max(matrix_t *m, float threshold, ivector2_t *peaks, int peaks_capacity)
{
  int num_peaks = 0;
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      float val = matrix_at(m, i, j);
      if (val < threshold) {
        continue; // below threshold
      }
      if ((i - 1 > 0 && matrix_at(m, i - 1, j) > val) ||
          (i + 1 < m->rows && matrix_at(m, i + 1, j) > val) ||
          (j - 1 > 0 && matrix_at(m, i, j - 1) > val) ||
          (j + 1 < m->cols && matrix_at(m, i, j + 1) > val)) {
        continue; // greater neighbor
      }
      peaks[num_peaks].i = i;
      peaks[num_peaks].j = j;
      num_peaks++;
      if (num_peaks == peaks_capacity) {
        return num_peaks;
      }
    }
  }
  return num_peaks;
}
