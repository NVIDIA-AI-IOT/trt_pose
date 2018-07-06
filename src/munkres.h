#pragma once

#include "tensor.h"
#include <cstring>

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
      return true; // go to procedure 4
    }
  }
  return false; // go to procedure 5
}

void munkres_step_4(bool *c0, bool *c1, bool *s, bool *p, int *p0, int *p1, int n, int m)
{
  while (true) {
    bool star = false;
    int s0 = 0;
    int s1 = 0;

    // search for star in prime's column
    for (int i = 0; i < n; i++) {
      if (s[IDX_2D(i, *p1, m)]) {
        star = true;
        s0 = i;
        s1 = *p1;
      }
    }

    s[IDX_2D(*p0, *p1, m)] = true; // star the prime
    if (!star) {
      break; // no star in column of prime, exit
    }
    
    s[IDX_2D(s0, s1, m)] = 0; // un-star the star in prime's column

    // search for prime in stars row
    for (int j = 0; j < m; j++) {
      // if prime found, update prime value
      if (p[IDX_2D(s0, j, m)]) {
        *p0 = s0;
        *p1 = j;
      }
    }
  }

  // clear primes and uncover lines
  for (int i = 0; i < n * m; i++) {
    p[i] = false;
  }
  for (int i = 0; i < n; i++) {
    c0[i] = false;
  }
  for (int j = 0; j < m; j++) {
    c1[j] = false;
  }
}

void munkres_step_5(float *a, bool *c0, bool *c1, int n, int m)
{
  float min = 1e9; // will break if larger

  for (int i = 0; i < n; i++) {
    if (c0[i]) {
      continue;
    }
    for (int j = 0; j < m; j++) {
      if (c1[j]) {
        continue;
      }
      if (a[IDX_2D(i, j, m)] < min) {
        min = a[IDX_2D(i, j, m)];
      }
    }
  }

  for (int i = 0; i < n; i++) {
    if (!c0[i]) {
      continue;
    }
    for (int j = 0; j < m; j++) {
      a[IDX_2D(i, j, m)] += min;
    }
  }

  for (int j = 0; j < m; j++) {
    if (c1[j]) {
      continue;
    }
    for (int i = 0; i < n; i++) {
      a[IDX_2D(i, j, m)] -= min;
    }
  }
}

void munkres(float *a, bool *c0, bool *c1, bool *s, bool *p, int n, int m)
{
  // preliminary
  int step = 0;
  if (m >= n) {
    munkres_sub_min_row(a, n, m);
  } 
  if (m > n) {
    step = 1;
  }

  bool done = false;
  int p0, p1;

  while (!done) {
    switch (step) {

      case 0: 
      {
        munkres_sub_min_col(a, n, m);
        step = 1;
        break;
      }

      case 1: 
      {
        munkres_step_1(a, s, n, m);
        step = 2;
        break;
      }

      case 2: 
      {
        if (munkres_step_2(s, c1, n, m)) {
          done = true;
        } else {
          step = 3;
        }
        break;
      }

      case 3: 
      {
        if (munkres_step_3(a, c0, c1, s, p, &p0, &p1, n, m)) {
          step = 4;
        } else {
          step = 5;
        }
        break;
      }

      case 4: 
      {
        munkres_step_4(c0, c1, s, p, &p0, &p1, n, m);
        step = 2;
        break;
      }

      case 5: 
      {
        munkres_step_5(a, c0, c1, n, m);
        step = 3;
        break;
      }
    }  
  }
}

size_t munkres_workspace_size(int n, int m)
{
  return sizeof(bool) * (n + m + n * m);
}

void munkres(float *a, bool *s, int n, int m, void *workspace, size_t workspace_size)
{
  bool *p = (bool *) workspace;
  bool *c0 = p + n * m;
  bool *c1 = c0 + n;
  memset(workspace, 0, workspace_size);
  munkres(a, c0, c1, s, p, n, m);
}
