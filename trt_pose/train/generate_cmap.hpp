#include <torch/extension.h>
#include <vector>
#include <cmath>

namespace trt_pose {
namespace train {

torch::Tensor generate_cmap(torch::Tensor counts, torch::Tensor peaks, int height, int width, float stdev, int window);

} // namespace trt_pose::train
} // namespace trt_pose
