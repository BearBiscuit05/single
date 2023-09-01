#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <set>
#include <utility>
#include <parallel_hashmap/phmap.h>
class PartitionEngine {
public:
    std::string graphPath;
    std::string srcPath;
    std::string dstPath;
    std::string trainMaskPath;
    std::ifstream srcStream;
    std::ifstream dstStream;
    std::ifstream trainIdStream;
    
    PartitionEngine();
    PartitionEngine(std::string graphPath);
    int readline(std::pair<int64_t, int64_t> &edge);
};


