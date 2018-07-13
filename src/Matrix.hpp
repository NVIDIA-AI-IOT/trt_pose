#pragma once

#include <memory>
#include <vector>
#include "tensor.h"

template<class T, class Alloc = std::allocator<T>>
class Matrix
{
public:
  Matrix(int nrows, int ncols) : nrows(nrows), ncols(ncols)
  {
    data = alloc.allocate(nrows * ncols);
    refcount = (int*) malloc(sizeof(int));
    *refcount = 1;
    shared = true;
  }

  Matrix(const Matrix<T> &other) : nrows(other.nrows), ncols(other.ncols)
  {
    if (other.shared)
    {
      shared = true;
      refcount = other.refcount;
      (*refcount)++;
    }
    else
    {
      shared = false;
      data = other.data;
      refcount = nullptr;
    }
  }

  Matrix(T *data, int nrows, int ncols) : data(data), nrows(nrows), ncols(ncols), shared(false) {};

  ~Matrix()
  {
    if (shared)
    {
      if (deref() == 0)
      {
        destroy();
      }
    }
  }

  int deref()
  {
    (*refcount)--;
    return *refcount;
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

  /**
   * Colmajor access
   */
  inline T at_c(int row, int col) const 
  {
    return data[IDX_2D_colmajor(row, col, ncols)];
  }

  inline T at_c_(int row, int col) const
  {
    return &data[IDX_2D_colmajor(row, col, ncols)];
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

  inline T addToCol(int col, T value)
  {
    for (int i = 0; i < nrows; i++) 
    {
      *at_(i, col) += value;
    }
  }

  inline T addToRow(int row, T value)
  {
    for (int j = 0; j < ncols; j++) 
    {
      *at_(row, j) += value;
    }
  }

  inline T sumIndices(const std::vector<std::pair<int, int>> &indices)
  {
    T sum = 0;
    for (int i = 0; i < indices.size(); i++)
    {
      sum += at(indices[i].first, indices[i].second);
    }
    return sum;
  }

  inline T copy(const Matrix<T> &other)
  {
    for (int i = 0; i < nrows; i++)
    {
      for (int j = 0; j < ncols; j++)
      {
        *at_(i, j) = other.at(i, j);
      }
    }
  }

  inline void fill(T value)
  {
    for (int i = 0; i < nrows; i++)
    {
      for (int j = 0; j < ncols; j++)
      {
        *at_(i, j) = value;
      }
    }
  }

  inline void fill_identity()
  {
    for (int i = 0; i < nrows; i++) {
      *at_(i, i) = 1.0f;
    }
  }

  /**
   * Frees memory and sets data pointer, resets ownership of data
   */
  inline void destroy()
  {
    if (shared)
    {
      alloc.deallocate(data, nrows * ncols);
      free(refcount);
    }
  }

  inline T* dataMutable()
  {
    return data;
  }

  const int nrows;
  const int ncols;

private:
  Alloc alloc;
  T *data;
  bool shared;
  int *refcount;
};
