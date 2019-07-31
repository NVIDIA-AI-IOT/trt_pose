#include "connect_parts.hpp"

std::unordered_map<int, int> searchPart(
    std::pair<int, int> node,
    const std::vector<PairGraph> &graphs,
    const std::vector<std::pair<int, int>> &topology,
    std::vector<std::vector<bool>> &visited)
{
  std::unordered_map<int, int> component;
  std::queue<std::pair<int, int>> queue;
  queue.push(node);

  while (!queue.empty())
  {
    auto node = queue.front();
    queue.pop();

    if (visited[node.first][node.second])
    {
      continue;
    }

    component.insert(node);
    visited[node.first][node.second] = true;

    for (size_t i = 0; i < graphs.size(); i++)
    {
      if (topology[i].first == node.first)
      {
        if (graphs[i].isRowSet(node.second))
        {
          queue.push({topology[i].second, graphs[i].colForRow(node.second)});
        }
      }
      else if (topology[i].second == node.first)
      {
        if (graphs[i].isColSet(node.second))
        {
          queue.push({topology[i].first, graphs[i].rowForCol(node.second)});
        }
      }
    }
  }
  return component;
}

// match 
std::vector<std::unordered_map<int, int>> connectParts(
    const std::vector<int> &part_counts,
    const std::vector<PairGraph> &graphs,
    const std::vector<std::pair<int, int>> &topology)
{
  std::vector<std::unordered_map<int, int>> components;
  std::vector<std::vector<bool>> visited(part_counts.size());
  for (size_t i = 0; i < part_counts.size(); i++)
  {
    visited[i].resize(part_counts[i]);
  }

  for (size_t i = 0; i < visited.size(); i++)
  {
    for (size_t j = 0; j < visited[i].size(); j++)
    {
      if (!visited[i][j])
      {
        auto comp = searchPart({i, j}, graphs, topology, visited);
        components.push_back(comp);
      } 
    }
  }

  return components;
};



// returns NxOxC -1 if part absent, 1 if present
std::vector<torch::Tensor> connect_parts(torch::Tensor score_graph, torch::Tensor topology, torch::Tensor counts, float score_threshold, int max_object_count)
{
    auto options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .layout(torch::kStrided)
        .device(torch::kCPU)
        .requires_grad(false);
    
    int N = counts.size(0);
    int C = counts.size(1);
    int K = topology.size(0);
    
    auto object_counts = torch::zeros({N}, options);
    auto object_counts_a = object_counts.accessor<int, 1>();
    auto objects = torch::full({N, max_object_count, C}, -1, options);
    auto objects_a = objects.accessor<int, 3>();
    auto cost_graph = -score_graph;
    auto topology_a = topology.accessor<int, 2>();
    auto counts_a = counts.accessor<int, 2>();
    auto cost_graph_a = cost_graph.accessor<float, 4>();
    
    // convert topology to vec
    std::vector<std::pair<int, int>> topology_vec(K);
    for (int k = 0; k < K; k++) {
        topology_vec[k] = {topology_a[k][2], topology_a[k][3]};
    }
    
    for (int n = 0; n < N; n++)
    {
        std::vector<PairGraph> pair_graphs;
        std::vector<int> part_counts;
        // get part connections
        for (int k = 0; k < K; k++)
        {
            int cmap_a_idx = topology_a[k][2];
            int cmap_b_idx = topology_a[k][3];
            int nrows = counts_a[n][cmap_a_idx];
            int ncols = counts_a[n][cmap_b_idx];
            auto pair_graph = PairGraph(nrows, ncols);
            auto cost_graph_a_nk = cost_graph_a[n][k];
            _munkres(cost_graph_a_nk, pair_graph, nrows, ncols);
            
            // remove pairs below score threshold
            for (int i = 0; i < pair_graph.nrows; i++)
            {
                for (int j = 0; j < pair_graph.ncols; j++)
                {
                    if (cost_graph_a_nk[i][j] < score_threshold && pair_graph.isPair(i, j)) {
                        pair_graph.reset(i, j);
                    }
                }
            }
            pair_graphs.push_back(pair_graph);
        }
        
        for (int c = 0; c < C; c++)
        {
            part_counts.push_back(counts_a[n][c]);
        }
        
        auto components = connectParts(part_counts, pair_graphs, topology_vec);
        size_t i;
        for (i = 0; i < components.size() && i < max_object_count; i++)
        {
            for (auto &p : components[i])
            {
              objects_a[n][i][p.first] = p.second;
            }
        }
        object_counts_a[n] = i;
    }
    return {objects, object_counts};
}
