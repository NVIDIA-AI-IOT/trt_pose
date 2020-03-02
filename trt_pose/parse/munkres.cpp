#include "munkres.hpp"
#include "utils/CoverTable.hpp"
#include "utils/PairGraph.hpp"

namespace trt_pose {
namespace parse {

using namespace utils;

void subMinRow(float *cost_graph, const int M, const int nrows,
               const int ncols) {
  for (int i = 0; i < nrows; i++) {
    // find min
    float min = cost_graph[i * M];
    for (int j = 0; j < ncols; j++) {
      float val = cost_graph[i * M + j];
      if (val < min) {
        min = val;
      }
    }

    // subtract min
    for (int j = 0; j < ncols; j++) {
      cost_graph[i * M + j] -= min;
    }
  }
}

void subMinCol(float *cost_graph, const int M, const int nrows,
               const int ncols) {
  for (int j = 0; j < ncols; j++) {
    // find min
    float min = cost_graph[j];
    for (int i = 0; i < nrows; i++) {
      float val = cost_graph[i * M + j];
      if (val < min) {
        min = val;
      }
    }

    // subtract min
    for (int i = 0; i < nrows; i++) {
      cost_graph[i * M + j] -= min;
    }
  }
}

void munkresStep1(const float *cost_graph, const int M, PairGraph &star_graph,
                  const int nrows, const int ncols) {
  for (int i = 0; i < nrows; i++) {
    for (int j = 0; j < ncols; j++) {
      if (!star_graph.isRowSet(i) && !star_graph.isColSet(j) &&
          (cost_graph[i * M + j] == 0)) {
        star_graph.set(i, j);
      }
    }
  }
}

// returns 1 if we should exit
bool munkresStep2(const PairGraph &star_graph, CoverTable &cover_table) {
  int k =
      star_graph.nrows < star_graph.ncols ? star_graph.nrows : star_graph.ncols;
  int count = 0;
  for (int j = 0; j < star_graph.ncols; j++) {
    if (star_graph.isColSet(j)) {
      cover_table.coverCol(j);
      count++;
    }
  }
  return count >= k;
}

bool munkresStep3(const float *cost_graph, const int M,
                  const PairGraph &star_graph, PairGraph &prime_graph,
                  CoverTable &cover_table, std::pair<int, int> &p,
                  const int nrows, const int ncols) {
  for (int i = 0; i < nrows; i++) {
    for (int j = 0; j < ncols; j++) {
      if (cost_graph[i * M + j] == 0 && !cover_table.isCovered(i, j)) {
        prime_graph.set(i, j);
        if (star_graph.isRowSet(i)) {
          cover_table.coverRow(i);
          cover_table.uncoverCol(star_graph.colForRow(i));
        } else {
          p.first = i;
          p.second = j;
          return 1;
        }
      }
    }
  }
  return 0;
};

void munkresStep4(PairGraph &star_graph, PairGraph &prime_graph,
                  CoverTable &cover_table, std::pair<int, int> p) {
  // repeat until no star found in prime's column
  while (star_graph.isColSet(p.second)) {
    // find and reset star in prime's column
    std::pair<int, int> s = {star_graph.rowForCol(p.second), p.second};
    star_graph.reset(s.first, s.second);

    // set this prime to a star
    star_graph.set(p.first, p.second);

    // repeat for prime in cleared star's row
    p = {s.first, prime_graph.colForRow(s.first)};
  }
  star_graph.set(p.first, p.second);
  cover_table.clear();
  prime_graph.clear();
}

void munkresStep5(float *cost_graph, const int M, const CoverTable &cover_table,
                  const int nrows, const int ncols) {
  bool valid = false;
  float min;
  for (int i = 0; i < nrows; i++) {
    for (int j = 0; j < ncols; j++) {
      if (!cover_table.isCovered(i, j)) {
        if (!valid) {
          min = cost_graph[i * M + j];
          valid = true;
        } else if (cost_graph[i * M + j] < min) {
          min = cost_graph[i * M + j];
        }
      }
    }
  }

  for (int i = 0; i < nrows; i++) {
    if (cover_table.isRowCovered(i)) {
      for (int j = 0; j < ncols; j++) {
        cost_graph[i * M + j] += min;
      }
      //       cost_graph.addToRow(i, min);
    }
  }
  for (int j = 0; j < ncols; j++) {
    if (!cover_table.isColCovered(j)) {
      for (int i = 0; i < nrows; i++) {
        cost_graph[i * M + j] -= min;
      }
      //       cost_graph.addToCol(j, -min);
    }
  }
}

void _munkres(float *cost_graph, const int M, PairGraph &star_graph,
              const int nrows, const int ncols) {
  PairGraph prime_graph(nrows, ncols);
  CoverTable cover_table(nrows, ncols);
  prime_graph.clear();
  cover_table.clear();
  star_graph.clear();

  int step = 0;
  if (ncols >= nrows) {
    subMinRow(cost_graph, M, nrows, ncols);
  }
  if (ncols > nrows) {
    step = 1;
  }

  std::pair<int, int> p;
  bool done = false;
  while (!done) {
    switch (step) {
    case 0:
      subMinCol(cost_graph, M, nrows, ncols);
    case 1:
      munkresStep1(cost_graph, M, star_graph, nrows, ncols);
    case 2:
      if (munkresStep2(star_graph, cover_table)) {
        done = true;
        break;
      }
    case 3:
      if (!munkresStep3(cost_graph, M, star_graph, prime_graph, cover_table, p,
                        nrows, ncols)) {
        step = 5;
        break;
      }
    case 4:
      munkresStep4(star_graph, prime_graph, cover_table, p);
      step = 2;
      break;
    case 5:
      munkresStep5(cost_graph, M, cover_table, nrows, ncols);
      step = 3;
      break;
    }
  }
}

std::size_t assignment_out_workspace(const int M) {
  return sizeof(float) * M * M;
}

void assignment_out(int *connections,         // 2xM
                    const float *score_graph, // MxM
                    const int count_a, const int count_b, const int M,
                    const float score_threshold, void *workspace) {
  const int nrows = count_a;
  const int ncols = count_b;

  // compute cost graph (negate score graph)
  float *cost_graph = (float *)workspace;
  for (int i = 0; i < count_a; i++) {
    for (int j = 0; j < count_b; j++) {
      const int idx = i * M + j;
      cost_graph[idx] = -score_graph[idx];
    }
  }

  // run munkres algorithm
  auto star_graph = PairGraph(nrows, ncols);
  _munkres(cost_graph, M, star_graph, nrows, ncols);

  // fill output connections
  for (int i = 0; i < nrows; i++) {
    for (int j = 0; j < ncols; j++) {
      if (star_graph.isPair(i, j) && score_graph[i * M + j] > score_threshold) {
        connections[0 * M + i] = j;
        connections[1 * M + j] = i;
      }
    }
  }
}

void assignment_out_k(int *connections,         // Kx2xM
                      const float *score_graph, // KxMxM
                      const int *topology,      // Kx4
                      const int *counts,        // C
                      const int K, const int M, const float score_threshold,
                      void *workspace) {
  for (int k = 0; k < K; k++) {
    const int *tk = &topology[k * 4];
    const int cmap_idx_a = tk[2];
    const int cmap_idx_b = tk[3];
    const int count_a = counts[cmap_idx_a];
    const int count_b = counts[cmap_idx_b];
    assignment_out(&connections[k * 2 * M], &score_graph[k * M * M], count_a,
                   count_b, M, score_threshold, workspace);
  }
}

void assignment_out_nk(int *connections,         // NxKx2xM
                       const float *score_graph, // NxKxMxM
                       const int *topology,      // Kx4
                       const int *counts,        // NxC
                       const int N, const int C, const int K, const int M,
                       const float score_threshold, void *workspace) {
  for (int n = 0; n < N; n++) {
    assignment_out_k(&connections[n * K * 2 * M], &score_graph[n * K * M * M],
                     topology, &counts[n * C], K, M, score_threshold,
                     workspace);
  }
}

} // namespace parse
} // namespace trt_pose
