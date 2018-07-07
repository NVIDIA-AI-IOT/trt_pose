#ifndef VECTOR_H
#define VECTOR_H

#include "tensor.h"
#include "math.h"

#ifdef __cplusplus
extern "C" {
#endif

// row major matrix
typedef struct vector2 {
  float i;
  float j;
} vector2_t;

typedef struct ivector2 {
  int i;
  int j;
} ivector2_t;

static inline float vector2_dot(vector2_t a, vector2_t b) {
  return a.i * b.i + a.j * b.j;
}

static inline float vector2_norm(vector2_t x) {
  return sqrtf(vector2_dot(x, x));
}

static inline vector2_t vector2_sub(vector2_t a, vector2_t b) {
  vector2_t c;
  c.i = a.i - b.i;
  c.j = a.j - b.j;
  return c;
}

static inline vector2_t vector2_add(vector2_t a, vector2_t b) {
  a.i += b.i;
  a.j += b.j;
  return a;
}

static inline vector2_t vector2_scale(vector2_t x, float scale) {
  x.i *= scale;
  x.j *= scale;
  return x; 
}

static inline vector2_t vector2_normalized(vector2_t x) {
  float norm = vector2_norm(x);
  return vector2_scale(x, 1.0f / norm);
}

static inline vector2_t vector2_add_scalar(vector2_t x, float scalar) {
  x.i += scalar;
  x.j += scalar;
  return x;
}

static inline vector2_t vector2_from_i(ivector2_t x) {
  vector2_t y;
  y.i = x.i;
  y.j = x.j;
  return y;
}

static inline ivector2_t peak_from_vector2(vector2_t x) {
  ivector2_t y;
  y.i = x.i;
  y.j = x.j;
  return y; 
}



#ifdef __cplusplus
}
#endif

#endif
