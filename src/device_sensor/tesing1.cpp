#include <iostream>
#include <fstream>
#include <cmath>

// 函数用于获取文件大小
long getFileSize(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return -1; // 返回 -1 表示文件打开失败
    }
    return file.tellg(); // 获取文件大小
}

int main() {
    std::string file_path = "your_file_path"; // 替换为您的文件路径

    // 获取文件大小
    long file_size = getFileSize(file_path);
    if (file_size == -1) {
        std::cerr << "Error opening file." << std::endl;
        return 1;
    }

    // 获取本机内存容量或用户指定的容量
    long max_memory_capacity = 1024 * 1024 * 1024; // 1 GB
    // 或者根据您的需要从用户输入中获取

    // 计算分区数，每个分区大小不超过内存容量的 70%
    double partition_size = static_cast<double>(max_memory_capacity) * 0.7;
    int num_partitions = std::ceil(static_cast<double>(file_size) / partition_size);

    std::cout << "File Size: " << file_size << " bytes" << std::endl;
    std::cout << "Max Memory Capacity: " << max_memory_capacity << " bytes" << std::endl;
    std::cout << "Partition Size: " << partition_size << " bytes" << std::endl;
    std::cout << "Number of Partitions: " << num_partitions << std::endl;

    // 在这里可以进行分区操作，根据需要读取文件内容并处理

    return 0;
}
