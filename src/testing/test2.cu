#include <cub/cub.cuh>
#include <iostream>

int main() {
    const int num_elements = 8;
    int h_input[num_elements] = {2, 3, 1, 4, 2, 5, 6, 3};

    int *d_input, *d_output, *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // 在GPU上分配内存
    cudaMalloc(&d_input, sizeof(int) * num_elements);
    cudaMalloc(&d_output, sizeof(int) * num_elements);

    // 将输入数据拷贝到GPU
    cudaMemcpy(d_input, h_input, sizeof(int) * num_elements, cudaMemcpyHostToDevice);

    // 查询所需的临时存储空间大小
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_input, d_output, num_elements);

    // 分配临时存储空间
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // 执行并行前缀和操作
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_input, d_output, num_elements);

    // 将计算结果从GPU拷贝回主机
    int h_output[num_elements];
    cudaMemcpy(h_output, d_output, sizeof(int) * num_elements, cudaMemcpyDeviceToHost);

    // 打印计算结果
    std::cout << "Input array: ";
    for (int i = 0; i < num_elements; ++i) {
        std::cout << h_input[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Prefix sum array: ";
    for (int i = 0; i < num_elements; ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    // 释放内存
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp_storage);

    return 0;
}
