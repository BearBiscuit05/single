#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <future>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdint>
#include <sys/types.h>
#include <fcntl.h>
#include <stdexcept>
#include <limits>

using namespace std;


int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <binfile_path> <num_nodes> <feat_len>" << std::endl;
        return 1;
    }

    // 获取命令行参数
    const char* binfilePath = argv[1];
    int numNodes = std::stoi(argv[2]);
    int featLen = std::stoi(argv[3]);

    // 在这里使用获取到的参数
    std::cout << "Binfile Path: " << binfilePath << std::endl;
    std::cout << "Number of Nodes: " << numNodes << std::endl;
    std::cout << "Feature Length: " << featLen << std::endl;
    // 3072441,10308445
    
    // int64_t NUM_NODE=41652230;
    // int featLen = 300;

    // int64_t NUM_NODE=77741046;
    // int featLen = 300;
    
    // int64_t NUM_NODE=105896555;
    // int featLen = 300;

    std::ofstream file(binfilePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << binfilePath << std::endl;
        return;
    }
    std::vector<int> featBlcok(featLen,0);
    for (int i = 0 ; i < numNodes ; i++)
        file.write(reinterpret_cast<const char*>(featBlcok.data()), featBlcok.size() * sizeof(int));
    
    
    file.close();
    std::cout << "Data has been written to " << binfilePath << std::endl;
    return 0;
}


