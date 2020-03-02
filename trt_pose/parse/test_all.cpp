#include <stdexcept>
#include "paf_score_graph.hpp"
#include "find_peaks.hpp"
#include "refine_peaks.hpp"
#include "munkres.hpp"

#define ABS(x) ((x) > 0 ? (x) : (-x))

using namespace trt_pose;
using namespace trt_pose::parse;

void test_find_peaks_out_hw()
{
  const int H = 4;
  const int W = 4;
  const int M = 10;
  const float threshold = 2.0;
  const int window_size = 3;

  int counts;
  int peaks[M * 2];
  const float input[H * W] = {

    0., 0., 0., 0.,
    0., 0., 3., 0.,
    0., 0., 0., 0.,
    1., 0., 0., 0.

  };

  find_peaks_out_hw(&counts, peaks, input, H, W, M, threshold, window_size);

  if (counts != 1) {
    throw std::runtime_error("Number of peaks should be 1.");
  }
  if (peaks[0] != 1) {
    throw std::runtime_error("Peak i coordinate should be 1.");
  }
  if (peaks[1] != 2) {
    throw std::runtime_error("Peak j coordinate should be 2,");
  }
}

void test_refined_peaks_out_hw()
{
  const int H = 4;
  const int W = 4;
  const int M = 1;
  const int window_size = 3;

  const int counts = 1;
  const int peaks[M * 2] = { 1, 2 };
  const float cmap[H * W] = {

    0., 0., 1., 0.,
    0., 2., 3., 1.,
    0., 0., 2., 0.,
    0., 0., 0., 0.

  };
  const float i_true = (0.5 + (1. * 0 + 2. * 1 + 3. * 1 + 1. * 1 + 2. * 2) / 9.) / H;
  const float j_true = (0.5 + (2. * 1 + 1. * 2 + 3. * 2 + 2. * 2 + 1. * 3) / 9.) / W;
  const float tolerance = 1e-5;

  float refined_peaks[M * 2];

  refine_peaks_out_hw(refined_peaks, &counts, peaks, cmap, H, W, M, window_size);

  if (ABS(refined_peaks[0] - i_true) > tolerance) {
    throw std::runtime_error("i coordinate incorrect");
  }
  if (ABS(refined_peaks[1] - j_true) > tolerance) {
    throw std::runtime_error("j coordinate incorrect");
  }
}

void test_paf_score_graph_hw()
{
  const int M = 2;
  const int H = 4;
  const int W = 4;
  const int counts_a = 2;
  const int counts_b = 2;
  const int num_integral_samples = 3;

  float score_graph[M * M];

  // test points
  //
  // _ _ _ b
  // _ _ _ |
  // a - b |
  // _ _ _ a
  
  const float paf_i[H * W] = {
    0., 0., 0., -1.,
    0., 0., 0., -1.,
    0., 0., 0., -1.,
    0., 0., 0., -1.
  };
  const float paf_j[H * W] = {
    0., 0., 0., 0.,
    0., 0., 0., 0.,
    1., 1., 1., 0.,
    0., 0., 0., 0.
  };
  const float peaks_a[M * 2] = {
    0.625, 0.125, // mid-left
    0.875, 0.875  // bot-right
  };
  const float peaks_b[M * 2] = {
    0.625, 0.625, // mid-mid
    0.125, 0.875  // top-right
  };
  
  paf_score_graph_out_hw(
      score_graph,
      paf_i,
      paf_j,
      counts_a,
      counts_b,
      peaks_a,
      peaks_b,
      H, W, M,
      num_integral_samples
      );
}

void test_assignment_out()
{
  const int M = 4;
  const int count_a = 3;
  const int count_b = 3;
  const float score_threshold = 0.3;
  
  std::size_t workspace_size = assignment_out_workspace(M);
  void *workspace = (void *) malloc(workspace_size);

  int connections[2 * M];
  const float score_graph[M * M] = {
    1., 3., 0., 0.,
    1., 2., 1., 0.,
    4., 3., 4., 0.,
    0., 0., 0., 0.,
  };

  assignment_out(connections, score_graph, count_a, count_b, M, score_threshold, workspace);

  if (connections[0] != 1) {
    throw std::runtime_error("connections[0] should be 1.");
  }
  if (connections[1] != 0) {
    throw std::runtime_error("connections[0] should be 1.");
  }
  if (connections[2] != 2) {
    throw std::runtime_error("connections[0] should be 1.");
  }

  free(workspace);
}


int main()
{
  test_find_peaks_out_hw();
  test_refined_peaks_out_hw();
  test_paf_score_graph_hw();
  test_assignment_out();
  return 0;
}
