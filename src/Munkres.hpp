#pragma once

#include "Matrix.hpp"
#include "PairGraph.hpp"
#include "CoverTable.hpp"

template<typename T>
void subMinRow(Matrix<T> &cost_graph)
{
  for (int i = 0; i < cost_graph.nrows; i++) 
  {
    cost_graph.addToRow(i, -cost_graph.minRow(i));
  }
}

template<typename T>
void subMinCol(Matrix<T> &cost_graph)
{
  for (int j = 0; j < cost_graph.ncols; j++)
  {
    cost_graph.addToCol(j, -cost_graph.minCol(j));
  }
}

template<typename T>
void munkresStep1(const Matrix<T> &cost_graph, PairGraph &star_graph)
{
  for (int i = 0; i < cost_graph.nrows; i++)
  {
    for (int j = 0; j < cost_graph.ncols; j++)
    {
      if (!star_graph.isRowSet(i) && !star_graph.isColSet(j))
      {
        star_graph.set(i, j);
      }
    }
  }
}

// returns 1 if we should exit
bool munkresStep2(const PairGraph &star_graph, CoverTable &cover_table)
{
  int k = star_graph.nrows < star_graph.ncols ? star_graph.nrows : star_graph.ncols;
  int count = 0;
  for (int j = 0; j < star_graph.ncols; j++) 
  {
    if (star_graph.isColSet(j)) 
    {
      cover_table.coverCol(j);
      count++;
    }
  }
  return count >= k;
}

template<typename T>
bool munkresStep3(const Matrix<T> &cost_graph, const PairGraph &star_graph, PairGraph &prime_graph, CoverTable &cover_table, std::pair<int, int> &p)
{
  for (int i = 0; i < cost_graph.nrows; i++)
  {
    for (int j = 0; j < cost_graph.ncols; j++)
    {
      if (cost_graph.at(i, j) == 0 && !cover_table.isCovered(i, j))
      {
        prime_graph.set(i, j);
        if (star_graph.isRowSet(i))
        {
          cover_table.coverRow(i);
          cover_table.uncoverCol(star_graph.colForRow(i));
        }
        else
        {
          p.first = i;
          p.second = j;
          return 1;
        }
      }
    }
  }
  return 0;
}; 

void munkresStep4(PairGraph &star_graph, PairGraph &prime_graph, CoverTable &cover_table, std::pair<int, int> p)
{
  // repeat until no star found in prime's column
  while (star_graph.isColSet(p.second))
  {
    // find and reset star in prime's column 
    std::pair<int, int> s = { star_graph.rowForCol(p.second), p.second }; 
    star_graph.reset(s.first, s.second);

    // set this prime to a star
    star_graph.set(p.first, p.second);

    // repeat for prime in cleared star's row
    p = { s.first, prime_graph.colForRow(s.first) };
  }

  cover_table.clear();
  prime_graph.clear();
}

template<typename T>
void munkresStep5(Matrix<T> &cost_graph, const CoverTable &cover_table)
{
  bool valid = false;
  T min;
  for (int i = 0; i < cost_graph.nrows; i++)
  {
    for (int j = 0; j < cost_graph.ncols; j++)
    {
      if (!cover_table.isCovered(i, j))
      {
        if (!valid)
        {
          min = cost_graph.at(i, j);
          valid = true;
        }
        else if (cost_graph.at(i, j) < min)
        {
          min = cost_graph.at(i, j);
        }
      }
    }
  }

  for (int i = 0; i < cost_graph.nrows; i++)
  {
    if (cover_table.isRowCovered(i))
    {
      cost_graph.addToRow(i, min);
    }
  }
  for (int j = 0; j < cost_graph.ncols; j++)
  {
    if (!cover_table.isColCovered(j))
    {
      cost_graph.addToCol(j, -min);
    }
  }
}

template<typename T>
void munkres(Matrix<T> &cost_graph, PairGraph &star_graph)
{
  PairGraph prime_graph(star_graph.nrows, star_graph.ncols);  
  CoverTable cover_table(star_graph.nrows, star_graph.ncols);
  prime_graph.clear();
  cover_table.clear();
  star_graph.clear();

  int step = 0;
  if (cost_graph.ncols >= cost_graph.nrows)
  {
    subMinRow(cost_graph);
  }
  if (cost_graph.ncols > cost_graph.nrows)
  {
    step = 1;
  }

  std::pair<int, int> p;
  bool done = false;
  while (!done)
  {
    switch(step)
    {
      case 0:
        subMinCol(cost_graph);
      case 1:
        munkresStep1(cost_graph, star_graph);
      case 2:
        if(munkresStep2(star_graph, cover_table))
        {
          done = true;
          break;
        }
      case 3:
        if (!munkresStep3(cost_graph, star_graph, prime_graph, cover_table, p))
        {
          step = 5;
          break;
        }
      case 4:
        munkresStep4(star_graph, prime_graph, cover_table, p);
        step = 2;
        break;
      case 5:
        munkresStep5(cost_graph, cover_table);
    }
  }
}
