#ifndef MUNKRES_H
#define MUNKRES_H

#include "string.h"

#ifdef __cplusplus
extern "C" {
#endif

void _munkres_sub_min_row(float *a, int n, int m);
void _munkres_sub_min_col(float *a, int n, int m);
void _munkres_step_1(float *a, int *s, int n, int m);
// returns 1 if finished, c1 should be initialized 0
int _munkres_step_2(int *s, int *c1, int n, int m);
int _munkres_step_3_all_covered(float *a, int *c0, int *c1, int n, int m);
// p0, p1 will contain the row, col of the prime (applicable if return is 1);
int _munkres_step_3_prime(float *a, int *c0, int *c1, int *s, int *p, int *p0, int *p1, int n, int m);
int _munkres_step_3(float *a, int *c0, int *c1, int *s, int *p, int *p0, int *p1, int n, int m);
void _munkres_step_4(int *c0, int *c1, int *s, int *p, int *p0, int *p1, int n, int m);
void _munkres_step_5(float *a, int *c0, int *c1, int n, int m);
void _munkres(float *a, int *c0, int *c1, int *s, int *p, int n, int m);

size_t munkres_workspace_size(int n, int m);
void munkres(float *a, int *s, int n, int m, void *workspace, size_t workspace_size);

#ifdef __cplusplus
}
#endif

#endif
