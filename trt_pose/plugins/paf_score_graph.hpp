#include <torch/extension.h>
#include <vector>


void paf_score_graph_out(torch::Tensor score_graph, torch::Tensor paf, torch::Tensor topology, torch::Tensor counts, torch::Tensor peaks, int num_integral_samples);
torch::Tensor paf_score_graph(torch::Tensor paf, torch::Tensor topology, torch::Tensor counts, torch::Tensor peaks, int num_integral_samples);