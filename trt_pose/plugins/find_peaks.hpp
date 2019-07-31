#include <torch/extension.h>
#include <vector>


void find_peaks_out(torch::Tensor counts, torch::Tensor peaks, torch::Tensor input, float threshold, int window_size, int max_count);
std::vector<torch::Tensor> find_peaks(torch::Tensor input, float threshold, int window_size, int max_count);