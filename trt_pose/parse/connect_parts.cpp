#include "connect_parts.hpp"
#include <queue>

namespace trt_pose {
namespace parse {

std::size_t connect_parts_out_workspace(const int C, const int M) {
  return sizeof(int) * C * M;
}

void connect_parts_out(int *object_counts,     // 1
                       int *objects,           // PxC
                       const int *connections, // Kx2xM
                       const int *topology,    // Kx4
                       const int *counts,      // C
                       const int K, const int C, const int M, const int P,
                       void *workspace) {

  // initialize objects
  for (int i = 0; i < C * M; i++) {
    objects[i] = -1;
  }

  // initialize visited
  std::memset(workspace, 0, connect_parts_out_workspace(C, M));
  int *visited = (int *)workspace;

  int num_objects = 0;

  for (int c = 0; c < C; c++) {
    if (num_objects >= P) {
      break;
    }

    const int count = counts[c];

    for (int i = 0; i < count; i++) {
      if (num_objects >= P) {
        break;
      }

      std::queue<std::pair<int, int>> q;
      bool new_object = false;
      q.push({c, i});

      while (!q.empty()) {
        auto node = q.front();
        q.pop();
        int c_n = node.first;
        int i_n = node.second;

        if (visited[c_n * M + i_n]) {
          continue;
        }

        visited[c_n * M + i_n] = 1;
        new_object = true;
        objects[num_objects * C + c_n] = i_n;

        for (int k = 0; k < K; k++) {
          const int *tk = &topology[k * 4];
          const int c_a = tk[2];
          const int c_b = tk[3];
          const int *ck = &connections[k * 2 * M];

          if (c_a == c_n) {
            int i_b = ck[i_n];
            if (i_b >= 0) {
              q.push({c_b, i_b});
            }
          }

          if (c_b == c_n) {
            int i_a = ck[M + i_n];
            if (i_a >= 0) {
              q.push({c_a, i_a});
            }
          }
        }
      }

      if (new_object) {
        num_objects++;
      }
    }
  }
  *object_counts = num_objects;
}

void connect_parts_out_batch(int *object_counts,     // N
                             int *objects,           // NxPxC
                             const int *connections, // NxKx2xM
                             const int *topology,    // Kx4
                             const int *counts,      // NxC
                             const int N, const int K, const int C, const int M,
                             const int P, void *workspace) {
  for (int n = 0; n < N; n++) {
    connect_parts_out(&object_counts[n], &objects[n * P * C],
                      &connections[n * K * 2 * M], topology, &counts[n * C], K,
                      C, M, P, workspace);
  }
}

} // namespace parse
} // namespace trt_pose
