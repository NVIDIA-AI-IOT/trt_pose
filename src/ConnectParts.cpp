#include "ConnectParts.hpp"

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
