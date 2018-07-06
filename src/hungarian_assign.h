#include "tensor.h"

int count_zeros_in_row(int *C, int N, int M, int i)
{
  int n = 0;
  for (int j = 0 ; j < M; j++) {
    if (C[IDX_2D(i, j, M)] == 0) {
      n++;
    }
  }
  return n;
}

int count_zeros_in_col(int *C, int N, int M, int j)
{
  int n = 0;
  for (int i = 0; i < N; i++) {
    if (C[IDX_2D(i, j, M)] == 0) {
      n++;
    }
  }
  return n;
}

int count_zeros(int *C, int N, int M) {
  int n = 0;
  for (int i = 0; i < N * M; i++) {
    if (C[i] == 0) {
      n++;
    }
  }
  return n;
}

int hungarian_first_zero_in_row(int *C, int N, int M, int i) {
  for (int j = 0; j < M; j++) {
    if (C[IDX_2D(i, j, M)] == 0) {
      return j;
    }
  }
  return -1;  // no zero found
}

int hungarian_first_zero_in_col(int *C, int N, int M, int j) {
  for (int i = 0; i < N; i++) {
    if (C[IDX_2D(i, j, M)] == 0) {
      return i;
    }
  }
  return -1;  // no zero found
}

void hungarian_cross_out_row(int *C, int N, int M, int i)
{
  for (int j = 0; j < M; j++) {
    C[IDX_2D(i, j, M)] = -1; 
  } 
}

void hungarian_cross_out_col(int *C, int N, int M, int j)
{
  for (int i = 0; i < N; i++) {
    C[IDX_2D(i, j, M)] = -1; 
  } 
}

bool hungarian_assign_single_zero_row(int *C, int N, int M) 
{
  // check for row with 1 zero
  for (int i = 0; i < N; i++) {
    int a_i = count_zeros_in_row(C, N, M, i);
    if (a_i == 1) {
      int j = hungarian_first_zero_in_row(C, N, M, i);
      hungarian_cross_out_row(C, N, M, i);
      hungarian_cross_out_col(C, N, M, j);
      C[IDX_2D(i, j, M)] = -2; // set assignee to -2
      return true;
    }
  }
  return false;
}

bool hungarian_assign_single_zero_col(int *C, int N, int M)
{
  // check for col with 1 zero
  for (int j = 0; j < M; j++) {
    int a_j = count_zeros_in_col(C, N, M, j);
    if (a_j == 1) {
      int i = hungarian_first_zero_in_col(C, N, M, j);
      hungarian_cross_out_row(C, N, M, i);
      hungarian_cross_out_col(C, N, M, j);
      C[IDX_2D(i, j, M)] = -2;
      return true;
    }
  }
  return false;
}

void hungarian_assign(int *C, int N, int M)
{
  int k = 0;
  while (k < N) {

    if (hungarian_assign_single_zero_row(C, N, M)) {
      k++;
      continue;
    }

    if (hungarian_assign_single_zero_col(C, N, M)) {
      k++;
      continue;
    }

    // no cols / rows with only 1 zero found, so pick first zero
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < M; j++) {
        if (C[IDX_2D(i, j, M)] == 0) {
          hungarian_cross_out_row(C, N, M, i);
          hungarian_cross_out_col(C, N, M, j);
          C[IDX_2D(i, j, M)] = -2;
          k++;
        }
      }
    }
  }
}

// check for optimality... M >= N
bool hungarian_check_optimality(int *C, int N, int M)
{
  int n = count_zeros(C, N, M);
  int k = 0;

  while (n > 0) {
    int a_max = 0;
    int dim_max = 0;
    int idx_max = 0;

    // check for row with max zeros
    for (int i = 0; i < N; i++) {
      int a_i = count_zeros_in_row(C, N, M, i);
      if (a_i > a_max) {
        a_max = a_i;
        dim_max = 0;
        idx_max = i;
      }
    }

    // check for col with max zeros
    for (int j = 0; j < M; j++) {
      int a_j = count_zeros_in_col(C, N, M, j);
      if (a_j > a_max) {
        a_max = a_j;
        dim_max = 1;
        idx_max = j;
      }
    }

    // cross off max row or column (by setting to nonzero value)
    if (dim_max == 0) {
      hungarian_cross_out_row(C, N, M, idx_max);
    } else {
      hungarian_cross_out_col(C, N, M, idx_max);
    }

    n -= a_max;
    k++;
  }

  return k >= N;
}

// for minimization, use sub_min_*, assumes G is >= 0
void hungarian_sub_min_row(float *G, int N, int M)
{
  for (int i = 0; i < N; i++) {
    float min = G[IDX_2D(i, 0, M)];
    for (int j = 0; j < M; j++) {
      float val = G[IDX_2D(i, j, M)];
      if (val < min) {
        min = val;
      }
    }
    for (int j = 0; j < M; j++) {
      G[IDX_2D(i, j, M)] -= min;
    }
  } 
}

void hungarian_sub_min_col(float *G, int N, int M)
{
  for (int j = 0; j < M; j++) {
    float min = G[IDX_2D(0, j, M)];
    for (int i = 0; i < N; i++) {
      float val = G[IDX_2D(i, j, M)];
      if (val < min) {
        min = val;
      }
    }
    for (int i = 0; i < N; i++) {
      G[IDX_2D(i, j, M)] -= min;
    }
  } 
}

void hungarian_generate_mask(float *G, int N, int M, int *C)
{
  for (int i = 0; i < N * M; i++) {
    C[i] = (G[i] != 0);
  }
}
