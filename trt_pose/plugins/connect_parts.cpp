#include "connect_parts.hpp"


void connect_parts_out(torch::Tensor object_counts, torch::Tensor objects, torch::Tensor connections, torch::Tensor topology, torch::Tensor counts, int max_count)
{
    auto options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .layout(torch::kStrided)
        .device(torch::kCPU)
        .requires_grad(false);
    int N = counts.size(0);
    int K = topology.size(0);
    int C = counts.size(1);
    int M = connections.size(3);
    
    auto visited = torch::zeros({N, C, M}, options);
    auto visited_a = visited.accessor<int, 3>();
    auto counts_a = counts.accessor<int, 2>();
    auto topology_a = topology.accessor<int, 2>();
    auto objects_a = objects.accessor<int, 3>();
    auto object_counts_a = object_counts.accessor<int, 1>();
    auto connections_a = connections.accessor<int, 4>();
    
    for (int n = 0; n < N; n++)
    {
        int num_objects = 0;
        for (int c = 0; c < C; c++)
        {
            if (num_objects >= max_count) {
                break;
            }
            
            int count = counts_a[n][c];
            
            for (int i = 0; i < count; i++)
            {
                if (num_objects >= max_count) {
                    break;
                }
                
                std::queue<std::pair<int, int>> q;
                bool new_object = false;
                q.push({c, i});
                
                while (!q.empty())
                {
                    auto node = q.front();
                    q.pop();
                    int c_n = node.first;
                    int i_n = node.second;
                    
                    if (visited_a[n][c_n][i_n]) {
                        continue;
                    }
                    
                    visited_a[n][c_n][i_n] = 1;
                    new_object = true;
                    objects_a[n][num_objects][c_n] = i_n;
                    
                    for (int k = 0; k < K; k++)
                    {
                        int c_a = topology_a[k][2];
                        int c_b = topology_a[k][3];
                        
                        if (c_a == c_n)
                        {
                            int i_b = connections_a[n][k][0][i_n];
                            if (i_b >= 0) {
                                q.push({c_b, i_b});
                            }
                        }
                        
                        if (c_b == c_n)
                        {
                            int i_a = connections_a[n][k][1][i_n];
                            if (i_a >= 0) {
                                q.push({c_a, i_a});
                            }
                        }
                    }
                }
                
                if (new_object)
                {
                    num_objects++;
                }
            }
        }
        
        object_counts_a[n] = num_objects;
    }
}


std::vector<torch::Tensor> connect_parts(torch::Tensor connections, torch::Tensor topology, torch::Tensor counts, int max_count)
{
    auto options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .layout(torch::kStrided)
        .device(torch::kCPU)
        .requires_grad(false);
    
    int N = counts.size(0);
    int K = topology.size(0);
    int C = counts.size(1);
    int M = connections.size(3);
    
    auto objects = torch::full({N, max_count, C}, -1, options);
    auto object_counts = torch::zeros({N}, options);
    connect_parts_out(object_counts, objects, connections, topology, counts, max_count);
    return {object_counts, objects};
}