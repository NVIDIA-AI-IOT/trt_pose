#pragma once

void refine_peaks_out_hw(
    float *refined_peaks, 
    const int *counts,
    const int *peaks,
    const float *cmap,
    const int H,
    const int W,
    const int M,
    const int window_size
);

void refine_peaks_out_chw(
    float *refined_peaks, 
    const int *counts,
    const int *peaks,
    const float *cmap,
    const int C,
    const int H,
    const int W,
    const int M,
    const int window_size
);


void refine_peaks_out_nchw(
    float *refined_peaks, 
    const int *counts,
    const int *peaks,
    const float *cmap,
    const int N,
    const int C,
    const int H,
    const int W,
    const int M,
    const int window_size
);
