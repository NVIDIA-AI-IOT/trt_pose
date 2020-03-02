#pragma once
#include <cstring>

namespace trt_pose {
namespace parse {

std::size_t connect_parts_out_workspace(const int C, const int M);

void connect_parts_out(int *object_counts,     // 1
                       int *objects,           // PxC
                       const int *connections, // Kx2xM
                       const int *topology,    // Kx4
                       const int *counts,      // C
                       const int K, const int C, const int M, const int P,
                       void *workspace);

void connect_parts_out_batch(int *object_counts,     // N
                             int *objects,           // NxPxC
                             const int *connections, // NxKx2xM
                             const int *topology,    // Kx4
                             const int *counts,      // NxC
                             const int N, const int K, const int C, const int M,
                             const int P, void *workspace);

} // namespace parse
} // namespace trt_pose
