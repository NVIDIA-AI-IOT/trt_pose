#pragma once

#include "Matrix.hpp"

void gaussNewtonUpdate(Matrix<float> &param, Matrix<float> &residual, Matrix<float> &jacobian);
