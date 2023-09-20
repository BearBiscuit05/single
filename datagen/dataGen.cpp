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


int main() {
    std::ofstream file(savePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << savePath << std::endl;
        return;
    }
    file.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(int));
    file.close();
    std::cout << "Data has been written to " << savePath << std::endl;
    return;
    return 0;
}
