#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include "PairGraph.hpp"
#include "Matrix.hpp"
#include "Part.hpp"
#include "Component.hpp"

Component ConnectedPartsSearch(
    int i, int j,
    const std::vector<std::vector<Part>> &parts,
    const std::vector<PairGraph> &graphs,
    const std::vector<std::pair<int, int>> &topology,
    std::vector<std::vector<bool>> &visited)
{
  Component component;
  std::queue<Part> queue;
  queue.push(parts[i][j]);

  while (!queue.empty())
  {
    auto node = queue.front();
    queue.pop();

    if (visited[node.channel][node.idx])
    {
      continue;
    }

    component.addPart(node);
    visited[node.channel][node.idx] = true;

    for (size_t i = 0; i < graphs.size(); i++)
    {
      if (topology[i].first == node.channel)
      {
        if (graphs[i].isRowSet(node.idx))
        {
          int channel = topology[i].second;
          int idx = graphs[i].colForRow(node.idx);
          queue.push(parts[channel][idx]);
        }
      }
      else if (topology[i].second == node.channel)
      {
        if (graphs[i].isColSet(node.idx))
        {
          int channel = topology[i].first;
          int idx = graphs[i].rowForCol(node.idx);
          queue.push(parts[channel][idx]);
        }
      }
    }
  }
  return component;
}

// match 
std::vector<Component> ConnectParts(
    const std::vector<std::vector<Part>> &parts,
    const std::vector<PairGraph> &graphs,
    const std::vector<std::pair<int, int>> &topology)
{
  std::vector<Component> components;
  std::vector<std::vector<bool>> visited(parts.size());
  for (size_t i = 0; i < parts.size(); i++)
  {
    visited[i].resize(parts[i].size());
  }

  for (size_t i = 0; i < visited.size(); i++)
  {
    for (size_t j = 0; j < visited[i].size(); j++)
    {
      if (!visited[i][j])
      {
        auto comp = ConnectedPartsSearch(i, j, parts, graphs, topology, visited);
        components.push_back(comp);
      } 
    }
  }

  return components;
};
