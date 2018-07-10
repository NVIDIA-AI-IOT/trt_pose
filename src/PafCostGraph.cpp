#include "PafCostGraph.hpp"

#define EPS 1e-6

Matrix<float> pafCostGraph(
    const std::pair<const Matrix<float> &, const Matrix<float> &> &paf,
    const std::pair<const std::vector<std::pair<int, int>>&, const std::vector<std::pair<int, int>>&> &peaks,
    int num_samples)
{
  Matrix<float> cost_graph(peaks.first.size(), peaks.second.size());
  for (size_t i = 0; i < peaks.first.size(); i++)
  {
    std::pair<float, float> p0 = peaks.first[i];
    p0.first += 0.5;
    p0.second += 0.5;
    for (size_t j = 0; j < peaks.second.size(); j++)
    {
      std::pair<float, float> p1 = peaks.second[j];
      p1.first += 0.5;
      p1.second += 0.5;
      std::pair<float, float> p01 = { p1.first - p0.first, p1.second - p0.second };
      float norm = sqrtf(p01.first * p01.first + p01.second * p01.second) + EPS; // numerical stability
      std::pair<float, float> p01_unit = { p01.first / norm, p01.second / norm };

      float sum = 0.0f;
      for (int n = 0; n < num_samples; n++)
      {
        std::pair<int, int> sample_idx = { (int) (p0.first + p01.first * (float) n / (float) num_samples),
                                (int) (p0.second + p01.second * (float) n / (float) num_samples) };
        std::pair<float, float> sample_val = { 
          paf.first.at(sample_idx.first, sample_idx.second),
          paf.second.at(sample_idx.first, sample_idx.second) 
        };

        float dot_product = sample_val.first * p01_unit.first + sample_val.second * p01_unit.second;
        sum += dot_product;
      }
      *cost_graph.at_(i, j) = - sum / num_samples;
    }
  }
  return cost_graph;
}

