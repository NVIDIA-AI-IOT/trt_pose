#include "generate_cmap.hpp"

namespace trt_pose {
namespace train {

torch::Tensor generate_cmap(torch::Tensor counts, torch::Tensor peaks, int height, int width, float stdev, int window)
{
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .layout(torch::kStrided)
        .device(torch::kCPU)
        .requires_grad(false);
    
    int N = peaks.size(0);
    int C = peaks.size(1);
    int M = peaks.size(2);
    int H = height;
    int W = width;
    int w = window / 2;
    
    auto cmap = torch::zeros({N, C, H, W}, options);
    auto cmap_a = cmap.accessor<float, 4>();
    auto counts_a = counts.accessor<int, 2>();
    auto peaks_a = peaks.accessor<float, 4>();
    float var = stdev * stdev;
    
    for (int n = 0; n < N; n++)
    {
        for (int c = 0; c < C; c++)
        {
            int count = counts_a[n][c];
            for (int p = 0; p < count; p++)
            {
                float i_mean = peaks_a[n][c][p][0] * H;
                float j_mean = peaks_a[n][c][p][1] * W;
                int i_min = i_mean - w;
                int i_max = i_mean + w + 1;
                int j_min = j_mean - w;
                int j_max = j_mean + w + 1;
                if (i_min < 0) i_min = 0;
                if (i_max >= H) i_max = H;
                if (j_min < 0) j_min = 0;
                if (j_max >= W) j_max = W;
                
                for (int i = i_min; i < i_max; i++)
                {
                    float d_i = i_mean - ((float) i + 0.5);
                    float val_i = - (d_i * d_i);
                    for (int j = j_min; j < j_max; j++)
                    {
                        float d_j = j_mean - ((float) j + 0.5);
                        float val_j = - (d_j * d_j);
                        float val_ij = val_i + val_j;
                        float val = expf(val_ij / var);
                        
                        if (val > cmap_a[n][c][i][j])
                        {
                            cmap_a[n][c][i][j] = val;
                        }
                    }
                }
            }
        }
    }
    
    return cmap;
}

} // namespace trt_pose::train
} // namespace trt_pose
