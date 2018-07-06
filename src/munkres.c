#include "munkres.h"
#include "tensor.h"

void _munkres_sub_min_row(float *a, int n, int m)
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

void _munkres_sub_min_col(float *a, int n, int m)
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

void _munkres_step_1(float *a, int *s, int n, int m)
{
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      if (a[IDX_2D(i, j, m)] == 0) {
        int star = 1; // whether to star this zero

        // check for star in column (if so, don't star)
        for (int k = 0; k < n; k++) {
          if (s[IDX_2D(k, j, m)]) {
            star = 0;
          }
        }

        // check for star in row (if so, don't star)
        for (int k = 0; k < m; k++) {
          if (s[IDX_2D(i, k, m)]) {
            star = 0;
          }
        }

        s[IDX_2D(i, j, m)] = star;
      }
    }
  }
}

// returns 1 if finished, c1 should be initialized 0
int _munkres_step_2(int *s, int *c1, int n, int m)
{
  int k = n < m ? n : m;
  int count = 0;

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      if (s[IDX_2D(i, j, m)]) {
        c1[j] = 1;
      }
    }
  }
  for (int j = 0; j < m; j++) {
    if (c1[j] == 1) {
      count++;
    }
  }
  return count >= k;
}

int _munkres_step_3_all_covered(float *a, int *c0, int *c1, int n, int m)
{
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      if (a[IDX_2D(i, j, m)] == 0 && !c0[i] && !c1[j]) {
        return 0;
      }
    }
  }
  return 1;
}

// p0, p1 will contain the row, col of the prime (applicable if return is 1)
int _munkres_step_3_prime(float *a, int *c0, int *c1, int *s, int *p, int *p0, int *p1, int n, int m)
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
        p[IDX_2D(i, j, m)] = 1;
        *p0 = i;
        *p1 = j;
        // cover row and uncover col of starred if exists
        for (int k = 0; k < m; k++) {
          if (s[IDX_2D(i, k, m)]) {
           c0[i] = 1;
           c1[k] = 0; 
           return 0; // remain in procedure 3
          }
        }
        return 1; // go to procedure 4
      }
    }
  }
  return 0; // remain in procedure 3 (really, go to 5)
}

int _munkres_step_3(float *a, int *c0, int *c1, int *s, int *p, int *p0, int *p1, int n, int m)
{
  while (!_munkres_step_3_all_covered(a, c0, c1, n, m)) 
  {
    if (_munkres_step_3_prime(a, c0, c1, s, p, p0, p1, n, m)) {
      return 1; // go to procedure 4
    }
  }
  return 0; // go to procedure 5
}

void _munkres_step_4(int *c0, int *c1, int *s, int *p, int *p0, int *p1, int n, int m)
{
  while (1) {
    int star = 0;
    int s0 = 0;
    int s1 = 0;

    // search for star in prime's column
    for (int i = 0; i < n; i++) {
      if (s[IDX_2D(i, *p1, m)]) {
        star = 1;
        s0 = i;
        s1 = *p1;
      }
    }

    s[IDX_2D(*p0, *p1, m)] = 1; // star the prime
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
    p[i] = 0;
  }
  for (int i = 0; i < n; i++) {
    c0[i] = 0;
  }
  for (int j = 0; j < m; j++) {
    c1[j] = 0;
  }
}

void _munkres_step_5(float *a, int *c0, int *c1, int n, int m)
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

void _munkres(float *a, int *c0, int *c1, int *s, int *p, int n, int m)
{
  // preliminary
  int step = 0;
  if (m >= n) {
    _munkres_sub_min_row(a, n, m);
  } 
  if (m > n) {
    step = 1;
  }

  int done = 0;
  int p0, p1;

  while (!done) {
    switch (step) {

      case 0: 
      {
        _munkres_sub_min_col(a, n, m);
        step = 1;
        break;
      }

      case 1: 
      {
        _munkres_step_1(a, s, n, m);
        step = 2;
        break;
      }

      case 2: 
      {
        if (_munkres_step_2(s, c1, n, m)) {
          done = 1;
        } else {
          step = 3;
        }
        break;
      }

      case 3: 
      {
        if (_munkres_step_3(a, c0, c1, s, p, &p0, &p1, n, m)) {
          step = 4;
        } else {
          step = 5;
        }
        break;
      }

      case 4: 
      {
        _munkres_step_4(c0, c1, s, p, &p0, &p1, n, m);
        step = 2;
        break;
      }

      case 5: 
      {
        _munkres_step_5(a, c0, c1, n, m);
        step = 3;
        break;
      }
    }  
  }
}

size_t munkres_workspace_size(int n, int m)
{
  return sizeof(int) * (n + m + n * m);
}

void munkres(float *a, int *s, int n, int m, void *workspace, size_t workspace_size)
{
  int *p = (int *) workspace;
  int *c0 = p + n * m;
  int *c1 = c0 + n;
  memset(workspace, 0, workspace_size);
  _munkres(a, c0, c1, s, p, n, m);
}
