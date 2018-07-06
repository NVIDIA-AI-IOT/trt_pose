#pragma once

#include "tensor.h"

void munkres_sub_min_row(float *a, int n, int m)
{
  for (int i = 0; i < n; i++) {
    float min = a[IDX_2D(i, 0, m)];
    for (int j = 1; j < m; j++) {
      if (a[IDX_2D(i, j, m)] < min) {
        min = a[IDX_2D(i, j, m)];
      }
    }
    for (int j = 0; j < m; j++) {
      a[IDX_2D(i, j, m)] -= min;
    }
  }
}

void munkres_sub_min_col(float *a, int n, int m)
{
  for (int j = 0; j < m; j++) {
    float min = a[IDX_2D(0, j, m)];
    for (int i = 1; i < n; i++) {
      if (a[IDX_2D(i, j, m)] < min) {
        min = a[IDX_2D(i, j, m)];
      }
    }
    for (int i = 0; i < n; i++) {
      a[IDX_2D(i, j, m)] -= min;
    }
  }
}

void munkres_step_1(float *a, bool *s, int n, int m)
{
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      if (a[IDX_2D(i, j, m)] == 0) {
        bool star = true; // whether to star this zero

        // check for star in column (if so, don't star)
        for (int k = 0; k < n; k++) {
          if (s[IDX_2D(k, j, m)]) {
            star = false;
          }
        }

        // check for star in row (if so, don't star)
        for (int k = 0; k < m; k++) {
          if (s[IDX_2D(i, k, m)]) {
            star = false;
          }
        }

        s[IDX_2D(i, j, m)] = star;
      }
    }
  }
}

// returns true if finished
bool munkres_step_2(bool *s, bool *c1, int n, int m)
{
  int k = n < m ? n : m;
  int count = 0;

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      if (s[IDX_2D(i, j, m)]) {
        c1[j] = true;
      }
    }
  }
  for (int j = 0; j < m; j++) {
    if (c1[j] == true) {
      count++;
    }
  }
  return count >= k;
}
