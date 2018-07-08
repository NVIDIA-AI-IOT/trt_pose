#pragma once

#include <array>

class PairGraph
{
public:

  PairGraph(int rows, int cols)
  {
    this->cols = cols;
    this->rows = rows;
    this->col = (int*) malloc(sizeof(int) * this->rows);
    this->row = (int*) malloc(sizeof(int) * this->cols);
  }

  ~PairGraph()
  {
    free(this->col);
    free(this->row);
  }
  
  /**
   * Returns the column index of the pair matching this row
   */
  inline int col_pair(int row)
  {
    return this->col[row];
  }

  /**
   * Returns the row index of the pair matching this column
   */
  inline int row_pair(int col)
  {
    return this->row[col];
  }

  /**
   * Creates a pair between row and col
   */
  inline void set(int row, int col)
  {
    this->col[row] = col;
    this->row[col] = row;
  }

  /**
   * Clears pair between row and col
   */
  inline void reset(int row, int col)
  {
    this->col[row] = -1;
    this->row[col] = -1;
  }

  /**
   * Clears all pairs in graph
   */
  void clear()
  {
    for (int i = 0; i < this->rows; i++) 
    {
      this->col[i] = -1;
    }
    for (int j = 0; j < this->cols; j++)
    {
      this->row[j] = -1;
    }
  }

private:
  int *row;
  int *col;
  int rows;
  int cols;
};
