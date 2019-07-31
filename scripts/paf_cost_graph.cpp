#include <torch/extension.h>
#include <vector>
#include <cmath>

#define EPS 1e-6

void paf_cost_graph_out(torch::Tensor cost_graph, torch::Tensor paf, torch::Tensor topology, torch::Tensor counts, torch::Tensor peaks, int num_integral_samples)
{
    int N = paf.size(0);
    int K = topology.size(0);
    int M = peaks.size(2);
    int H = paf.size(2);
    int W = paf.size(3);
    
    auto cost_graph_a = cost_graph.accessor<float, 4>();
    auto paf_a = paf.accessor<float, 4>();
    auto topology_a = topology.accessor<int, 2>();
    auto counts_acc = counts.accessor<int, 2>();
    auto peaks_acc = peaks.accessor<float, 4>();
    
    for (int n = 0; n < N; n++)
    {
        for (int k = 0; k < K; k++)
        {
            auto cost_graph_nk = cost_graph_a[n][k];
            auto paf_i_idx = topology_a[k][0];
            auto paf_j_idx = topology_a[k][1];
            auto cmap_a_idx = topology_a[k][2];
            auto cmap_b_idx = topology_a[k][3];
            auto paf_i = paf_a[n][paf_i_idx];
            auto paf_j = paf_a[n][paf_j_idx];
            
            auto counts_a = counts_acc[n][cmap_a_idx];
            auto counts_b = counts_acc[n][cmap_b_idx];
            auto peaks_a = peaks_acc[n][cmap_a_idx];
            auto peaks_b = peaks_acc[n][cmap_b_idx];
            
            for (int a = 0; a < counts_a; a++)
            {
                // compute point A
                float pa_i = peaks_a[a][0] * H;
                float pa_j = peaks_a[a][1] * W;
                
                for (int b = 0; b < counts_b; b++)
                {
                    // compute point B
                    float pb_i = peaks_b[b][0] * H;
                    float pb_j = peaks_b[b][1] * W;
                        
                    // compute vector A->B
                    float pab_i = pb_i - pa_i;
                    float pab_j = pb_j - pa_j;
                    
                    // compute normalized vector A->B
                    float pab_norm = sqrtf(pab_i * pab_i + pab_j * pab_j) + EPS;
                    float uab_i = pab_i / pab_norm;
                    float uab_j = pab_j / pab_norm;
                        
                    float integral = 0.0;
                    float progress = 0.0;
                    float increment = 1.0f / num_integral_samples;
                    
                    for (int t = 0; t < num_integral_samples; t++)
                    {
                        // compute integral point T
                        float progress = (float) t / (float) num_integral_samples;
                        float pt_i = pa_i + progress * pab_i; //(1.0 - progress) * pa_i + progress * pb_i;
                        float pt_j = pa_j + progress * pab_j;//(1.0 - progress) * pa_j + progress * pb_j;
                        
                        // convert to int
                        int pt_i_int = (int) pt_i;
                        int pt_j_int = (int) pt_j;
                        
                        // skip point if out of bounds (will weaken integral)
                        if (pt_i_int < 0) continue;
                        if (pt_i_int > H) continue;
                        if (pt_j_int < 0) continue;
                        if (pt_j_int > W) continue;
                        
                        // get vector at integral point from PAF
                        float pt_paf_i = paf_i[pt_i_int][pt_j_int];
                        float pt_paf_j = paf_j[pt_i_int][pt_j_int];
                        
                        // compute dot product of normalized A->B with PAF vector at integral point
                        float dot = pt_paf_i * uab_i + pt_paf_j * uab_j;
                        integral += dot;
                        
                        progress += increment;
                    }
                    
                    // normalize integral by number of samples
                    integral /= num_integral_samples;
                    cost_graph_nk[a][b] = integral;
                }
            }
        }
    }
}

// paf = Nx(2*K)xHxW
// topology = Kx4 --> (paf_i_idx, paf_j_idx, cmap_a_idx, cmap_b_idx)
// counts = NxC
// peaks = NxCxMx2
// cost_graph = NxKxMxM

torch::Tensor paf_cost_graph(torch::Tensor paf, torch::Tensor topology, torch::Tensor counts, torch::Tensor peaks, int num_integral_samples)
{
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .layout(torch::kStrided)
        .device(torch::kCPU)
        .requires_grad(false);
    
    int N = peaks.size(0);
    int K = topology.size(0);
    int M = peaks.size(2);
    
    auto cost_graph = torch::zeros({N, K, M, M}, options);
    paf_cost_graph_out(cost_graph, paf, topology, counts, peaks, num_integral_samples);
    return cost_graph;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &paf_cost_graph, "paf_cost_graph forward");
}