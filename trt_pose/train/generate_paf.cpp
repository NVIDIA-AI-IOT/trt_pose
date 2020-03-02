#include "generate_paf.hpp"

#define EPS 1e-5;

namespace trt_pose {
namespace train {

torch::Tensor generate_paf(torch::Tensor connections, torch::Tensor topology, torch::Tensor counts, torch::Tensor peaks, int height, int width, float stdev)
{
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .layout(torch::kStrided)
        .device(torch::kCPU)
        .requires_grad(false);
    
    int N = connections.size(0);
    int K = topology.size(0);
    int H = height;
    int W = width;
//     int C = counts.size(1);
    
    auto connections_a = connections.accessor<int, 4>();
    auto counts_a = counts.accessor<int, 2>();
    auto peaks_a = peaks.accessor<float, 4>();
    auto topology_a = topology.accessor<int, 2>();
    
    auto paf = torch::zeros({N, 2 * K, H, W}, options);
    auto paf_a = paf.accessor<float, 4>();
    float var = stdev * stdev;
    
    for (int n = 0; n < N; n++)
    {
        for (int k = 0; k < K; k++)
        {
            int k_i = topology_a[k][0];
            int k_j = topology_a[k][1];
            int c_a = topology_a[k][2];
            int c_b = topology_a[k][3];
            int count = counts_a[n][c_a];
            
            for (int i = 0; i < H; i++)
            {
                for (int j = 0; j < W; j++)
                {
                    float p_c_i = i + 0.5;
                    float p_c_j = j + 0.5;
                    
                    for (int i_a = 0; i_a < count; i_a++)
                    {
                        int i_b = connections_a[n][k][0][i_a];

                        if (i_b < 0) {
                            continue; // connection doesn't exist
                        }

                        auto p_a = peaks_a[n][c_a][i_a];
                        auto p_b = peaks_a[n][c_b][i_b];

                        float p_a_i = p_a[0] * H;
                        float p_a_j = p_a[1] * W;
                        float p_b_i = p_b[0] * H;
                        float p_b_j = p_b[1] * W;
                        float p_ab_i = p_b_i - p_a_i;
                        float p_ab_j = p_b_j - p_a_j;
                        float p_ab_mag = sqrtf(p_ab_i * p_ab_i + p_ab_j * p_ab_j) + EPS;
                        float u_ab_i = p_ab_i / p_ab_mag;
                        float u_ab_j = p_ab_j / p_ab_mag;
                
                
                        float p_ac_i = p_c_i - p_a_i;
                        float p_ac_j = p_c_j - p_a_j;
                        
                        // dot product to find tangent bounds
                        float dot = p_ac_i * u_ab_i + p_ac_j * u_ab_j;
                        float tandist = 0.0;
                        if (dot < 0.0) {
                            tandist = dot;
                        }
                        else if (dot > p_ab_mag) {
                            tandist = dot - p_ab_mag;
                        }
                        
                        // cross product to find perpendicular bounds
                        float cross = p_ac_i * u_ab_j - p_ac_j * u_ab_i;
                        
                        // scale exponentially RBF by 2D distance from nearest point on line segment
                        float scale = expf(-(tandist*tandist + cross*cross) / var);
                        paf_a[n][k_i][i][j] += scale * u_ab_i;
                        paf_a[n][k_j][i][j] += scale * u_ab_j;
                    }
                }
            }
        }
    }
    
    return paf;
}

} // namespace trt_pose::train
} // namespace trt_pose
