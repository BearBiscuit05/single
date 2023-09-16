#include <iostream>
#include <fstream>
#include <sstream>
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
class ReadEngine {
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
    
    size_t readSize = 4096 * 16;
    int batch = readSize / sizeof(int64_t);
    off_t chunkSize = 0;
    off_t offset = 0;

    ReadEngine();
    ReadEngine(std::string graphPath);
    int readline(std::pair<int64_t, int64_t> &edge);
    void loadingMmapBlock();
    void unmapBlock(int64_t* addr, off_t size);
    void readTrainIdx(std::vector<int64_t>& ids);
    int readlines(std::vector<std::pair<int64_t, int64_t>> &edges,std::vector<int64_t>& eids,int& edgesNUM);
};



class TGEngine{
public:
    std::string graphPath;

    int Fd;
    off_t edgeLength;
    int* edgeAddr;

    int edgeNUM=0;
    int nodeNUM=0;
    int readPtr=0;
    std::vector<int> degrees;
    size_t readSize = 4096 * 16;
    int batch = readSize / sizeof(int);
    off_t chunkSize = 0;
    off_t offset = 0;

    TGEngine();
    TGEngine(int nodeNUM,int edgeNUM);
    TGEngine(std::string graphPath,int nodeNUM,int edgeNUM);
    void loadingMmapBlock();
    void unmapBlock(int* addr, off_t size);
    int readline(std::pair<int, int> &edge);
    void convert2bin(std::string raw_graphPath,std::string new_graphPath,char delimiter,bool saveDegree,std::string degreePath);
    void readDegree(std::string degreePath,std::vector<int>& degreeList);
};
