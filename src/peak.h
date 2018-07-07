#ifndef PEAK_H
#define PEAK_H

#include "vector.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct peak {
  int row;
  int col;
} peak_t;

static inline vector2_t peak_to_vector2(peak_t peak) {
  vector2_t x;
  x.i = peak.row;
  x.j = peak.col;
  return x;
}

static inline peak_t peak_from_vector2(vector2_t x) {
  peak_t peak;
  peak.row = x.i;
  peak.col = x.j;
  return peak; 
}

#ifdef __cplusplus
}
#endif

#endif
