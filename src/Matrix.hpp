#pragma once

#include "tensor.h"

template<typename T>
class Matrix
{
public:
  Matrix(T *data, int rows, int cols) : data(data), rows(rows), cols(cols) {};

  /**
   * Access element at row, col
   */
  inline T at(int row, int col)
  {
    return data[IDX_2D(row, col, cols)];
  }

  /**
   * Mutable access at row, col
   */
  inline T *at_(int row, int col)
  {
    return &data[IDX_2D(row, col, cols)];
  }

private:
  T *data;
  int rows;
  int cols;
};
