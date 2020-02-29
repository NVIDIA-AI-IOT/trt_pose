#include "connect_parts.hpp"
#include "find_peaks.hpp"
#include "generate_cmap.hpp"
#include "generate_paf.hpp"
#include "munkres.hpp"
#include "paf_score_graph.hpp"
#include "refine_peaks.hpp"
#include <torch/extension.h>
#include <vector>

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("find_peaks", &find_peaks_torch, "find_peaks");
  m.def("find_peaks_out", &find_peaks_out_torch, "find_peaks_out");
  m.def("paf_score_graph", &paf_score_graph, "paf_score_graph");
  m.def("paf_score_graph_out", &paf_score_graph_out, "paf_score_graph_out");
  m.def("refine_peaks", &refine_peaks, "refine_peaks");
  m.def("refine_peaks_out", &refine_peaks_out, "refine_peaks_out");
  m.def("munkres", &munkres, "munkres");
  m.def("connect_parts", &connect_parts, "connect_parts");
  m.def("assignment", &assignment, "assignment");
  m.def("assignment_out", &assignment_out, "assignment_out");
  m.def("generate_cmap", &generate_cmap, "generate_cmap");
  m.def("generate_paf", &generate_paf, "generate_paf");
}
