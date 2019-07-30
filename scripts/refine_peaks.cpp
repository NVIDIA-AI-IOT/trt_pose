#include <torch/extension.h>
#include <vector>

void refine_peaks_out(torch::Tensor refined_peaks, torch::Tensor counts, torch::Tensor peaks, torch::Tensor cmap, int window_size)
{
    auto refined_peaks_a = refined_peaks.accessor<float, 4>();
    auto counts_a = counts.accessor<int, 2>();
    auto peaks_a = peaks.accessor<int, 4>();
    auto cmap_a = cmap.accessor<float, 4>();
    
    int w = window_size / 2;
    int width = cmap.size(3);
    int height = cmap.size(2);
    
    for (int b = 0; b < cmap.size(0); b++)
    {
        for (int c = 0; c < cmap.size(1); c++)
        {
            int count = counts_a[b][c];
            auto refined_peaks_a_bc = refined_peaks_a[b][c];
            auto peaks_a_bc = peaks_a[b][c];
            auto cmap_a_bc = cmap_a[b][c];
            
            for (int p = 0; p < count; p++)
            {
                auto refined_peak = refined_peaks_a_bc[p];
                auto peak = peaks_a_bc[p];
                
                int i = peak[0];
                int j = peak[1];                
                float weight_sum = 0.0f;
                
                for (int ii = i - w; ii < i + w + 1; ii++)
                {
                    int ii_idx = ii;
                    
                    // reflect index at border
                    if (ii < 0) ii_idx = -ii;
                    else if (ii >= height) ii_idx = height - (ii - height) - 2;
                        
                    for (int jj = j - w; jj < j + w + 1; jj++)
                    {
                        int jj_idx = jj;

                        // reflect index at border
                        if (jj < 0) jj_idx = -jj;
                        else if (jj >= width) jj_idx = width - (jj - width) - 2;
                        
                        float weight = cmap_a_bc[ii_idx][jj_idx];
                        refined_peak[0] += weight * ii;
                        refined_peak[1] += weight * jj;
                        weight_sum += weight;
                    }
                }
                
                refined_peak[0] /= weight_sum;
                refined_peak[1] /= weight_sum;
                refined_peak[0] += 0.5;
                refined_peak[1] += 0.5;
                refined_peak[0] /= height;
                refined_peak[1] /= width;
            }
        }
    }
}

torch::Tensor refine_peaks(torch::Tensor counts, torch::Tensor peaks, torch::Tensor cmap, int window_size)
{
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .layout(torch::kStrided)
        .device(torch::kCPU)
        .requires_grad(false);
    
    auto refined_peaks = torch::zeros({peaks.size(0), peaks.size(1), peaks.size(2), peaks.size(3)}, options);
    refine_peaks_out(refined_peaks, counts, peaks, cmap, window_size);
    return refined_peaks;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &refine_peaks, "refine_peaks forward");
}