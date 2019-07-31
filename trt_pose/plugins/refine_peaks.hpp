#include <torch/extension.h>
#include <vector>


void refine_peaks_out(torch::Tensor refined_peaks, torch::Tensor counts, torch::Tensor peaks, torch::Tensor cmap, int window_size);
torch::Tensor refine_peaks(torch::Tensor counts, torch::Tensor peaks, torch::Tensor cmap, int window_size);