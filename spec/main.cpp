#include "readGraph.h"
#include "iostream"
#include <omp.h>
#define MAX_ITEMS 5000000    // 设置最大元素个数
#define P_ERROR 0.000001     // 设置误差

#define BATCH 1024
#define THREADNUM 8
int main() {
    // std::string graphPath = "/raid/bear/papers_bin/";
    // ReadEngine readengine(graphPath);
    // std::pair<int64_t,int64_t> edge(-1,-1);
    // int sum = 0;
    // while( -1 != readengine.readline(edge)){
    //     sum++;
    // }
    // std::cout << "one hop num :" << sum << std::endl;
    // std::cout << "all edges num :" << readengine.readPtr << std::endl;
    //std::string graphPath = "/home/bear/workspace/singleGNN/spec/edges.bin";
    // TGEngine tgEngine(graphPath,9498,153138);
    // int sum = 0;
    // std::pair<int,int> edge(-1,-1);
    // while( -1 != tgEngine.readline(edge)){
    //     std::cout << edge.first << " --> " << edge.second << std::endl;
    //     sum++;
    // }
    // std::cout << "one hop num :" << sum << std::endl;
    std::string inputfile = "/home/dzz/graphdataset/small.txt";
    std::string test="test.bin";
    TGEngine tgEngine(8,6);
    char delimiter = ' ';
    tgEngine.convert2bin(inputfile,test,delimiter,true,"degree.bin");
    std::vector<int> degree;
    tgEngine.readDegree("degree.bin",degree);
    return 0;
}
