#pragma once

#include "cusolverDn.h"
#include "matrix.h"

int matrix_solve_c_workspace_size( cusolverDnHandle_t handle, float *a_data, matrix_t *a_mat);
;

// must populate b_mat with identity matrix
// matrix must be symmetric positive-definite
int matrix_solve_c( cusolverDnHandle_t handle, float *a_data, matrix_t *a_mat,
    float *b_data, matrix_t *b_mat, 
    float *workspace, int workspace_size);
