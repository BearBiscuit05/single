#include "bloomfilter.h"
#include "readGraph.h"

#define MAX_ITEMS 5000000    // 设置最大元素个数
#define P_ERROR 0.000001     // 设置误差


int main() {
    static BaseBloomFilter stBloomFilter = {0};

    std::string graphPath = "/raid/bear/papers_bin/";
    ReadEngine readengine(graphPath);
    std::vector<int64_t> tids;
    readengine.readTrainIdx(tids);

    InitBloomFilter(&stBloomFilter, 0, tids.size(), P_ERROR);
    
    if(0 != BloomFilter_AddNodes(&stBloomFilter,tids)){
		std::cout << "add nodes failed " << std::endl;
        exit(0);
	}

    std::pair<int64_t,int64_t> edge;
    // std::vector<std::pair<int64_t,int64_t>> ChooseEdges;
    std::vector<int64_t> ChooseEdgesId;
    std::vector<int64_t> ChooseNodes;
    while( -1 != readengine.readline(edge)) {
        int ret = BloomFilter_CheckEdge(&stBloomFilter,edge);
        if(ret == 1)
        {
            ChooseNodes.emplace_back(edge.second);
            ChooseEdgesId.emplace_back(readengine.readPtr - 1);
        }
        else if(ret == 2)
        {
            ChooseNodes.emplace_back(edge.first);
            ChooseEdgesId.emplace_back(readengine.readPtr - 1);
        }
    }
    std::cout << "one hop num :" << ChooseEdgesId.size() << std::endl;
    std::cout << "all edges num :" << readengine.readPtr << std::endl;
    return 0;
}