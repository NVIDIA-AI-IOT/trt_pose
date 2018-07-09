#pragma once

#include <cmath>
#include "Matrix.hpp"

Matrix<float> pafCostGraph(
    const std::pair<const Matrix<float> &, const Matrix<float> &> &paf,
    const std::pair<const std::vector<std::pair<int, int>>&, const std::vector<std::pair<int, int>>&> &peaks,
    int num_samples);
