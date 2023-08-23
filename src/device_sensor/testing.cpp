#include <iostream>
#include <unistd.h>
#include <pthread.h>
#include <fstream>
#include <iomanip> // 用于设置输出格式

// 函数用于获取文件大小
long getFileSize(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return -1; // 返回 -1 表示文件打开失败
    }
    return file.tellg(); // 获取文件大小
}

int main(int argc, char* argv[]) {
    // 获取最大内存
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    long max_memory = pages * page_size;

    // 获取最大线程数
    pthread_attr_t attr;
    size_t stacksize;
    pthread_attr_init(&attr);
    pthread_attr_getstacksize(&attr, &stacksize);

    // 检查命令行参数是否包含文件路径
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <file_path>" << std::endl;
        return 1;
    }

    std::string file_path = argv[1];

    // 获取文件大小
    long file_size = getFileSize(file_path);

    if (file_size == -1) {
        std::cerr << "Error opening file." << std::endl;
        return 1;
    }

    std::cout << "Max Memory: " << max_memory / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Max Threads Stack Size: " << stacksize << " bytes" << std::endl;
    std::cout << "File Size: " << std::fixed << std::setprecision(2) << static_cast<double>(file_size) / (1024 * 1024) << " MB" << std::endl;

    return 0;
}
