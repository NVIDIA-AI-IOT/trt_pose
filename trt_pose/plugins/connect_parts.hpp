#include <torch/extension.h>
#include <vector>
#include <queue>


void connect_parts_out(torch::Tensor object_counts, torch::Tensor objects, torch::Tensor connections, torch::Tensor topology, torch::Tensor counts, int max_count);
std::vector<torch::Tensor> connect_parts(torch::Tensor connections, torch::Tensor topology, torch::Tensor counts, int max_count);