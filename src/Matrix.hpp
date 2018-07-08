#pragma once

#include <memory>
#include <vector>
#include "tensor.h"

template<typename T>
class Matrix
{
public:
  Matrix(int nrows, int ncols) : nrows(nrows), ncols(ncols)
  {
    this->alloc();
  }
  Matrix(T *data, int nrows, int ncols) : data(data), nrows(nrows), ncols(ncols), owning(false) {};
  ~Matrix()
  {
    if (owning)
    {
      this->destroy();
    }
  }

  /**
   * Access element at row, col
   */
  inline T at(int row, int col) const
  {
    return data[IDX_2D(row, col, ncols)];
  }

  /**
   * Mutable access at row, col
   */
  inline T *at_(int row, int col)
  {
    return &data[IDX_2D(row, col, ncols)];
  }

  inline T minRow(int row) const
  {
    T min = at(row, 0);
    for (int j = 0; j < ncols; j++)
    {
      if (at(row, j) < min)
      {
        min = at(row, j);
      }
    }
    return min;
  }

  inline T minCol(int col) const
  {
    T min = at(col, 0);
    for (int i = 0; i < nrows; i++)
    {
      if (at(i, col) < min)
      {
        min = at(i, col);
      }
    }
    return min;
  }

  T addToCol(int col, T value)
  {
    for (int i = 0; i < nrows; i++) 
    {
      *at_(i, col) += value;
    }
  }

  T addToRow(int row, T value)
  {
    for (int j = 0; j < ncols; j++) 
    {
      *at_(row, j) += value;
    }
  }

  T sumIndices(const std::vector<std::pair<int, int>> &indices)
  {
    T sum = 0;
    for (int i = 0; i < indices.size(); i++)
    {
      sum += at(indices[i].first, indices[i].second);
    }
    return sum;
  }

  T copy(const Matrix<T> &other)
  {
    for (int i = 0; i < nrows; i++)
    {
      for (int j = 0; j < ncols; j++)
      {
        *at_(i, j) = other.at(i, j);
      }
    }
  }

  /**
   * Allocates memory and sets data pointer, and sets ownership of data
   */
  inline void alloc()
  {
    data = (T*) malloc(sizeof(T) * nrows * ncols); 
    owning = true;
  }

  /**
   * Frees memory and sets data pointer, resets ownership of data
   */
  inline void destroy()
  {
    free(data);
    owning = false;
  }

  const int nrows;
  const int ncols;

private:
  T *data;
  bool owning;
};
