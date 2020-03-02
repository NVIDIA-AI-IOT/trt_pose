#include "parse/connect_parts.hpp"
#include "parse/find_peaks.hpp"
#include "parse/munkres.hpp"
#include "parse/paf_score_graph.hpp"
#include "parse/refine_peaks.hpp"
#include "train/generate_cmap.hpp"
#include "train/generate_paf.hpp"
#include <torch/extension.h>
#include <vector>

using namespace trt_pose::parse;
using namespace trt_pose::train;

void find_peaks_out_torch(torch::Tensor counts, torch::Tensor peaks,
                          torch::Tensor input, const float threshold,
                          const int window_size, const int max_count) {
  const int N = input.size(0);
  const int C = input.size(1);
  const int H = input.size(2);
  const int W = input.size(3);
  const int M = max_count;

  // get pointers to tensor data
  int *counts_ptr = (int *)counts.data_ptr();
  int *peaks_ptr = (int *)peaks.data_ptr();
  const float *input_ptr = (const float *)input.data_ptr();

  // find peaks
  find_peaks_out_nchw(counts_ptr, peaks_ptr, input_ptr, N, C, H, W, M,
                      threshold, window_size);
}

std::vector<torch::Tensor> find_peaks_torch(torch::Tensor input,
                                            const float threshold,
                                            const int window_size,
                                            const int max_count) {
  auto options = torch::TensorOptions()
                     .dtype(torch::kInt32)
                     .layout(torch::kStrided)
                     .device(torch::kCPU)
                     .requires_grad(false);

  const int N = input.size(0);
  const int C = input.size(1);
  const int H = input.size(2);
  const int W = input.size(3);
  const int M = max_count;

  // create output tensors
  auto counts = torch::zeros({N, C}, options);
  auto peaks = torch::zeros({N, C, M, 2}, options);

  // find peaks
  find_peaks_out_torch(counts, peaks, input, threshold, window_size, max_count);

  return {counts, peaks};
}

void refine_peaks_out_torch(torch::Tensor refined_peaks, torch::Tensor counts,
                            torch::Tensor peaks, torch::Tensor cmap,
                            int window_size) {
  const int N = cmap.size(0);
  const int C = cmap.size(1);
  const int H = cmap.size(2);
  const int W = cmap.size(3);
  const int M = peaks.size(2);

  refine_peaks_out_nchw(
      (float *)refined_peaks.data_ptr(), (const int *)counts.data_ptr(),
      (const int *)peaks.data_ptr(), (const float *)cmap.data_ptr(), N, C, H, W,
      M, window_size);
}

torch::Tensor refine_peaks_torch(torch::Tensor counts, torch::Tensor peaks,
                                 torch::Tensor cmap, int window_size) {
  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat32)
                     .layout(torch::kStrided)
                     .device(torch::kCPU)
                     .requires_grad(false);

  auto refined_peaks = torch::zeros(
      {peaks.size(0), peaks.size(1), peaks.size(2), peaks.size(3)}, options);
  refine_peaks_out_torch(refined_peaks, counts, peaks, cmap, window_size);
  return refined_peaks;
}

void paf_score_graph_out_torch(torch::Tensor score_graph, torch::Tensor paf,
                               torch::Tensor topology, torch::Tensor counts,
                               torch::Tensor peaks,
                               const int num_integral_samples) {
  const int N = paf.size(0);
  const int K = topology.size(0);
  const int C = peaks.size(1);
  const int H = paf.size(2);
  const int W = paf.size(3);
  const int M = score_graph.size(3);

  paf_score_graph_out_nkhw(
      (float *)score_graph.data_ptr(), (const int *)topology.data_ptr(),
      (const float *)paf.data_ptr(), (const int *)counts.data_ptr(),
      (const float *)peaks.data_ptr(), N, K, C, H, W, M, num_integral_samples);
}

torch::Tensor paf_score_graph_torch(torch::Tensor paf, torch::Tensor topology,
                                    torch::Tensor counts, torch::Tensor peaks,
                                    const int num_integral_samples) {
  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat32)
                     .layout(torch::kStrided)
                     .device(torch::kCPU)
                     .requires_grad(false);
  const int N = peaks.size(0);
  const int K = topology.size(0);
  const int M = peaks.size(2);

  torch::Tensor score_graph = torch::zeros({N, K, M, M}, options);
  paf_score_graph_out_torch(score_graph, paf, topology, counts, peaks,
                            num_integral_samples);
  return score_graph;
}

void assignment_out_torch(torch::Tensor connections, torch::Tensor score_graph,
                          torch::Tensor topology, torch::Tensor counts,
                          const float score_threshold) {
  const int N = counts.size(0);
  const int C = counts.size(1);
  const int K = topology.size(0);
  const int M = score_graph.size(2);
  void *workspace = (void *)malloc(assignment_out_workspace(M));

  assignment_out_nk(
      (int *)connections.data_ptr(), (const float *)score_graph.data_ptr(),
      (const int *)topology.data_ptr(), (const int *)counts.data_ptr(), N, C, K, M,
      score_threshold, workspace);

  free(workspace);
}

torch::Tensor assignment_torch(torch::Tensor score_graph,
                               torch::Tensor topology, torch::Tensor counts,
                               float score_threshold) {
  auto options = torch::TensorOptions()
                     .dtype(torch::kInt32)
                     .layout(torch::kStrided)
                     .device(torch::kCPU)
                     .requires_grad(false);

  int N = counts.size(0);
  int K = topology.size(0);
  int M = score_graph.size(2);

  auto connections = torch::full({N, K, 2, M}, -1, options);
  assignment_out_torch(connections, score_graph, topology, counts,
                       score_threshold);
  return connections;
}

void connect_parts_out_torch(torch::Tensor object_counts, torch::Tensor objects, torch::Tensor connections, torch::Tensor topology, torch::Tensor counts, int max_count)
{
  const int N = object_counts.size(0);
  const int K = topology.size(0);
  const int C = counts.size(1);
  const int M = connections.size(3);
  const int P = max_count;
  void *workspace = malloc(connect_parts_out_workspace(C, M));
  connect_parts_out_batch(
      (int *) object_counts.data_ptr(),
      (int *) objects.data_ptr(),
      (const int *) connections.data_ptr(),
      (const int *) topology.data_ptr(),
      (const int *) counts.data_ptr(),
      N, K, C, M, P, workspace);
  free(workspace);
}

std::vector<torch::Tensor> connect_parts_torch(torch::Tensor connections, torch::Tensor topology, torch::Tensor counts, int max_count)
{
    auto options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .layout(torch::kStrided)
        .device(torch::kCPU)
        .requires_grad(false);
    
    int N = counts.size(0);
    int K = topology.size(0);
    int C = counts.size(1);
    int M = connections.size(3);
    
    auto objects = torch::full({N, max_count, C}, -1, options);
    auto object_counts = torch::zeros({N}, options);
    connect_parts_out_torch(object_counts, objects, connections, topology, counts, max_count);
    return {object_counts, objects};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("find_peaks", &find_peaks_torch, "find_peaks");
  m.def("find_peaks_out", &find_peaks_out_torch, "find_peaks_out");
  m.def("paf_score_graph", &paf_score_graph_torch, "paf_score_graph");
  m.def("paf_score_graph_out", &paf_score_graph_out_torch,
        "paf_score_graph_out");
  m.def("refine_peaks", &refine_peaks_torch, "refine_peaks");
  m.def("refine_peaks_out", &refine_peaks_out_torch, "refine_peaks_out");
  // m.def("munkres", &munkres, "munkres");
  m.def("connect_parts", &connect_parts_torch, "connect_parts");
  m.def("connect_parts_out", &connect_parts_out_torch, "connect_parts_out");
  m.def("assignment", &assignment_torch, "assignment");
  m.def("assignment_out", &assignment_out_torch, "assignment_out");
  m.def("generate_cmap", &generate_cmap, "generate_cmap");
  m.def("generate_paf", &generate_paf, "generate_paf");
}
