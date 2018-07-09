#pragma once

#include <vector>
#include "Matrix.hpp"

template<class T>
std::vector<std::pair<int, int>> findPeaks(const Matrix<T> &m, T minval);
