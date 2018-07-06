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

// returns true if finished, c1 should be initialized 0
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

bool munkres_step_3_all_covered(float *a, bool *c0, bool *c1, int n, int m)
{
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      if (a[IDX_2D(i, j, m)] == 0 && !c0[i] && !c1[j]) {
        return false;
      }
    }
  }
  return true;
}

// p0, p1 will contain the row, col of the prime (applicable if return is true)
bool munkres_step_3_prime(float *a, bool *c0, bool *c1, bool *s, bool *p, int *p0, int *p1, int n, int m)
{
  for (int i = 0; i < n; i++) {
    if (c0[i]) {
      continue;
    }
    for (int j = 0; j < m; j++) {
      if (c1[j]) {
        continue;
      }
      if (a[IDX_2D(i, j, m)] == 0) {
        p[IDX_2D(i, j, m)] = true;
        *p0 = i;
        *p1 = j;
        // cover row and uncover col of starred if exists
        for (int k = 0; k < m; k++) {
          if (s[IDX_2D(i, k, m)]) {
           c0[i] = true;
           c1[k] = false; 
           return false; // remain in procedure 3
          }
        }
        return true; // go to procedure 4
      }
    }
  }
  return false; // remain in procedure 3 (really, go to 5)
}

bool munkres_step_3(float *a, bool *c0, bool *c1, bool *s, bool *p, int *p0, int *p1, int n, int m)
{
  while (!munkres_step_3_all_covered(a, c0, c1, n, m)) 
  {
    if (munkres_step_3_prime(a, c0, c1, s, p, p0, p1, n, m)) {
      return true;
    }
  }
  return false;
}
