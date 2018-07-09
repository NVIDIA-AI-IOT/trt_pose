#pragma once

#include <vector>
#include "Matrix.hpp"

template<class T>
std::vector<std::pair<int, int>> findPeaks(const Matrix<T> &m, T minval)
{
  std::vector<std::pair<int, int>> peaks;
  for (int i = 0; i < m.nrows; i++)
  {
    for (int j = 0; j < m.ncols; j++)
    {
      T val = m.at(i, j);
      if ((val < minval) ||
         (i - 1 > 0 && m.at(i - 1, j) > val) ||
         (i + 1 < m.nrows && m.at(i + 1, j) > val) ||
         (j - 1 > 0 && m.at(i, j - 1) > val) ||
         (j + 1 < m.ncols && m.at(i, j + 1) > val))
      {
        continue;
      }
      peaks.push_back({i, j});
    }
  }
  return peaks;
}
