#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <set>
#include <utility>
#include <parallel_hashmap/phmap.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdint>
#include <sys/types.h>
#include <fcntl.h>
class PartitionEngine {
public:
    std::string graphPath;
    std::string srcPath;
    std::string dstPath;
    std::string trainMaskPath;

    int srcFd;
    int dstFd;
    int tidsFd;
    off_t srcLength;
    off_t dstLength;
    off_t tidsLength;
    int64_t* srcAddr;
    int64_t* dstAddr;
    int64_t* tidsAddr;

    int64_t edgeNUM;
    int64_t readPtr=0;
    size_t readSize = 4096;
    std::vector<int64_t> srcCache;
    std::vector<int64_t> dstCache;
    PartitionEngine();
    PartitionEngine(std::string graphPath);
    int readline(std::pair<int64_t, int64_t> &edge);
};


