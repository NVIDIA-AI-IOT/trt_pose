#include <torch/extension.h>
#include <vector>
#include <cmath>


torch::Tensor generate_paf(torch::Tensor connections, torch::Tensor topology, torch::Tensor counts, torch::Tensor peaks, int height, int width, float stdev);