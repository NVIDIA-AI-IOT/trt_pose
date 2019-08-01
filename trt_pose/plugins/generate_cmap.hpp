#include <torch/extension.h>
#include <vector>
#include <cmath>


torch::Tensor generate_cmap(torch::Tensor counts, torch::Tensor peaks, int height, int width, float stdev, int window);