#pragma once

#include "Matrix.hpp"

// uses row major for simplicity, col-major is native for lapack so maybe update
void gaussNewtonUpdate(Matrix<float> param, Matrix<float> residual, Matrix<float> jacobian);
