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
    std::string graphPath = "/home/dzz/graphdataset/com-orkut/Dcom-orkut.ungraph.txt";
    std::string bin_graphPath = "/home/bear/workspace/singleGNN/spec/S5P/edge.bin";
    TGEngine tgEngine(bin_graphPath,3072441,10308445);
    std::pair<int,int> edge(-1,-1);
    char delimiter = '\t';
    tgEngine.convert2bin(graphPath,bin_graphPath,delimiter,false,"");
    // int sum = 0;
    
    // while( -1 != tgEngine.readline(edge)){
    // }
    // std::cout << "all edges num :" << tgEngine.readPtr << std::endl;
    // std::cout << "one hop num :" << sum << std::endl;
    // std::string inputfile = "/home/dzz/graphdataset/small.txt";
    // std::string test="test.bin";
    // TGEngine tgEngine(8,6);
    // char delimiter = ' ';
    // tgEngine.convert2bin(inputfile,test,delimiter,true,"degree.bin");
    // std::vector<int> degree;
    // tgEngine.readDegree("degree.bin",degree);
    // return 0;  
}
