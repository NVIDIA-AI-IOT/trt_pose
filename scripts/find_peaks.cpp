#include <torch/extension.h>
#include <vector>

void find_peaks_out(torch::Tensor output, torch::Tensor input, float threshold, int window_size, int max_count)
{
    auto output_a = output.accessor<int, 4>();
    auto input_a = input.accessor<float, 4>();
    
    int w = window_size / 2;
    int width = input.size(3);
    int height = input.size(2);
    
    for (int b = 0; b < input.size(0); b++)
    {
        for (int c = 0; c < input.size(1); c++)
        {
            int count = 0;
            auto output_a_bc = output_a[b][c];
            auto input_a_bc = input_a[b][c];
            
            for (int i = 0; i < height && count < max_count; i++)
            {
                for (int j = 0; j < width && count < max_count; j++)
                {
                    float value = input_a_bc[i][j];
                    
                    if (value < threshold)
                        continue;
                    
                    int ii_min = i - w;
                    int jj_min = j - w;
                    int ii_max = i + w + 1;
                    int jj_max = j + w + 1;
                    
                    if (ii_min < 0) ii_min = 0;
                    if (ii_max > height) ii_max = height;
                    if (jj_min < 0) jj_min = 0;
                    if (jj_max > width) jj_max = width;
                    
                    // get max
                    bool is_peak = true;
                    for (int ii = ii_min; ii < ii_max; ii++)
                    {
                        for (int jj = jj_min; jj < jj_max; jj++)
                        {
                            if (input_a_bc[ii][jj] > value) {
                                is_peak = false;
                            }
                        }
                    }
                    
                    if (is_peak) {
                        output_a_bc[count][0] = 1;
                        output_a_bc[count][1] = i;
                        output_a_bc[count][2] = j;
                        count++;
                    }
                }
            }
        }
    }
}

torch::Tensor find_peaks(torch::Tensor input, float threshold, int window_size, int max_count)
{
    auto options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .layout(torch::kStrided)
        .device(torch::kCPU)
        .requires_grad(false);
    
    auto output = torch::zeros({input.size(0), input.size(1), max_count, 3}, options); // valid, i, j
    find_peaks_out(output, input, threshold, window_size, max_count);
    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &find_peaks, "find_peaks forward");
}