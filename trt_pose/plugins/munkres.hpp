#include <torch/extension.h>
#include <vector>
#include "utils/PairGraph.hpp"


void _munkres(torch::TensorAccessor<float, 2> cost_graph, PairGraph &star_graph, int nrows, int ncols);
void munkres_out(torch::Tensor cost_graph_out, torch::Tensor cost_graph, torch::Tensor topology, torch::Tensor counts);
torch::Tensor munkres(torch::Tensor cost_graph, torch::Tensor topology, torch::Tensor counts);

// assignment NxKx2xM
void assignment_out(torch::Tensor connections, torch::Tensor score_graph, torch::Tensor topology, torch::Tensor counts, float score_threshold);
torch::Tensor assignment(torch::Tensor score_graph, torch::Tensor topology, torch::Tensor counts, float score_threshold);