#include "utils/PairGraph.hpp"
#include "utils/CoverTable.hpp"
#include "munkres.hpp"


void subMinRow(torch::TensorAccessor<float, 2> cost_graph, int nrows, int ncols)
{
  for (int i = 0; i < nrows; i++) 
  {
    // find min
    float min = cost_graph[i][0];
    for (int j = 0; j < ncols; j++) {
        float val = cost_graph[i][j];
        if (val < min) {
            min = val;
        }
    }
    
    // subtract min
    for (int j = 0; j < ncols; j++) {
        cost_graph[i][j] -= min;
    }
  }
}

void subMinCol(torch::TensorAccessor<float, 2> cost_graph, int nrows, int ncols)
{
  for (int j = 0; j < ncols; j++)
  {
    // find min
    float min = cost_graph[0][j];
    for (int i = 0; i < nrows; i++) {
        float val = cost_graph[i][j];
        if (val < min) {
            min = val;
        }
    }
    
    // subtract min
    for (int i = 0; i < nrows; i++) {
        cost_graph[i][j] -= min;
    }
  }
}

void munkresStep1(torch::TensorAccessor<float, 2> cost_graph, PairGraph &star_graph, int nrows, int ncols)
{
  for (int i = 0; i < nrows; i++)
  {
    for (int j = 0; j < ncols; j++)
    {
      if (!star_graph.isRowSet(i) && !star_graph.isColSet(j) && (cost_graph[i][j] == 0))
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

bool munkresStep3(torch::TensorAccessor<float, 2> cost_graph, const PairGraph &star_graph, PairGraph &prime_graph, CoverTable &cover_table, std::pair<int, int> &p, int nrows, int ncols)
{
  for (int i = 0; i < nrows; i++)
  {
    for (int j = 0; j < ncols; j++)
    {
      if (cost_graph[i][j] == 0 && !cover_table.isCovered(i, j))
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
  star_graph.set(p.first, p.second);
  cover_table.clear();
  prime_graph.clear();
}

void munkresStep5(torch::TensorAccessor<float, 2> cost_graph, const CoverTable &cover_table, int nrows, int ncols)
{
  bool valid = false;
  float min;
  for (int i = 0; i < nrows; i++)
  {
    for (int j = 0; j < ncols; j++)
    {
      if (!cover_table.isCovered(i, j))
      {
        if (!valid)
        {
          min = cost_graph[i][j];
          valid = true;
        }
        else if (cost_graph[i][j] < min)
        {
          min = cost_graph[i][j];
        }
      }
    }
  }

  for (int i = 0; i < nrows; i++)
  {
    if (cover_table.isRowCovered(i))
    {
      for (int j = 0; j < ncols; j++) {
          cost_graph[i][j] += min;
      }
//       cost_graph.addToRow(i, min);
    }
  }
  for (int j = 0; j < ncols; j++)
  {
    if (!cover_table.isColCovered(j))
    {
      for (int i = 0; i < nrows; i++) {
          cost_graph[i][j] -= min;
      }
//       cost_graph.addToCol(j, -min);
    }
  }
}


void _munkres(torch::TensorAccessor<float, 2> cost_graph, PairGraph &star_graph, int nrows, int ncols)
{
  PairGraph prime_graph(nrows, ncols);  
  CoverTable cover_table(nrows, ncols);
  prime_graph.clear();
  cover_table.clear();
  star_graph.clear();
    
  int step = 0;
  if (ncols >= nrows)
  {
    subMinRow(cost_graph, nrows, ncols);
  }
  if (ncols > nrows)
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
        subMinCol(cost_graph, nrows, ncols);
      case 1:
        munkresStep1(cost_graph, star_graph, nrows, ncols);
      case 2:
        if(munkresStep2(star_graph, cover_table))
        {
          done = true;
          break;
        }
      case 3:
        if (!munkresStep3(cost_graph, star_graph, prime_graph, cover_table, p, nrows, ncols))
        {
          step = 5;
          break;
        }
      case 4:
        munkresStep4(star_graph, prime_graph, cover_table, p);
        step = 2;
        break;
      case 5:
        munkresStep5(cost_graph, cover_table, nrows, ncols);
        step = 3;
        break;
    }
  }
}


void munkres_out(torch::Tensor cost_graph_out, torch::Tensor cost_graph, torch::Tensor topology, torch::Tensor counts)
{
    int N = counts.size(0);
    int K = topology.size(0);
    
    cost_graph_out.copy_(cost_graph);
    auto topology_a = topology.accessor<int, 2>();
    auto counts_a = counts.accessor<int, 2>();
    auto cost_graph_out_a = cost_graph_out.accessor<float, 4>();
    
    for (int n = 0; n < N; n++)
    {
        for (int k = 0; k < K; k++)
        {
            int cmap_a_idx = topology_a[k][2];
            int cmap_b_idx = topology_a[k][3];
            int nrows = counts_a[n][cmap_a_idx];
            int ncols = counts_a[n][cmap_b_idx];
            auto star_graph = PairGraph(nrows, ncols);
            _munkres(cost_graph_out_a[n][k], star_graph, nrows, ncols);
        }
    }
}

torch::Tensor munkres(torch::Tensor cost_graph, torch::Tensor topology, torch::Tensor counts)
{
    auto cost_graph_out = torch::empty_like(cost_graph);
    munkres_out(cost_graph_out, cost_graph, topology, counts);
    return cost_graph_out;
}


// assignment NxKx2xM
void assignment_out(torch::Tensor connections, torch::Tensor score_graph, torch::Tensor topology, torch::Tensor counts, float score_threshold)
{
    int N = counts.size(0);
    int K = topology.size(0);
    
    auto cost_graph = -score_graph;
    auto score_graph_a = score_graph.accessor<float, 4>();
    auto connections_a = connections.accessor<int, 4>();
    auto topology_a = topology.accessor<int, 2>();
    auto counts_a = counts.accessor<int, 2>();
    auto cost_graph_out_a = cost_graph.accessor<float, 4>();
    
    for (int n = 0; n < N; n++)
    {
        for (int k = 0; k < K; k++)
        {
            int cmap_a_idx = topology_a[k][2];
            int cmap_b_idx = topology_a[k][3];
            int nrows = counts_a[n][cmap_a_idx];
            int ncols = counts_a[n][cmap_b_idx];
            auto star_graph = PairGraph(nrows, ncols);
            auto cost_graph_out_a_nk = cost_graph_out_a[n][k];
            _munkres(cost_graph_out_a_nk, star_graph, nrows, ncols);
            
            auto connections_a_nk = connections_a[n][k];
            auto score_graph_a_nk = score_graph_a[n][k];
            
            for (int i = 0; i < nrows; i++) {
                for (int j = 0; j < ncols; j++) {
                    if (star_graph.isPair(i, j) && score_graph_a_nk[i][j] > score_threshold) {
                        connections_a_nk[0][i] = j;
                        connections_a_nk[1][j] = i;
                    }
                }
            }
        }
    }
}

torch::Tensor assignment(torch::Tensor score_graph, torch::Tensor topology, torch::Tensor counts, float score_threshold)
{
    auto options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .layout(torch::kStrided)
        .device(torch::kCPU)
        .requires_grad(false);
    
    int N = counts.size(0);
    int K = topology.size(0);
    int M = score_graph.size(2);
    
    auto connections = torch::full({N, K, 2, M}, -1, options);
    assignment_out(connections, score_graph, topology, counts, score_threshold);
    return connections;
}