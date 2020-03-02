#include <torch/extension.h>
#include <vector>
#include <cmath>

namespace trt_pose {
namespace train {

torch::Tensor generate_paf(torch::Tensor connections, torch::Tensor topology, torch::Tensor counts, torch::Tensor peaks, int height, int width, float stdev);

} // namespace trt_pose::train
} // namespace trt_pose
