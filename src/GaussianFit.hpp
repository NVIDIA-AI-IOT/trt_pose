#pragma once

#include "GaussNewton.hpp"
#include "Gaussian.hpp"
#include "Matrix.hpp"

#define NPARAM 4
#define DEFAULT_ALPHA 1.0f
#define DEFAULT_SIGMA2 1.0f

void computeResidualJacobian( const std::pair<int, int> &peak, const Matrix<float> &cmap, const Matrix<float> &param,
    Matrix<float> &residual, Matrix<float> &jacobian, int wsize);

Gaussian gaussianFit(const std::pair<int, int> &peak, const Matrix<float> &cmap, int wsize, int niter);
