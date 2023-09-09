#include "bloomfilter.h"
#include "readGraph.h"
#include <omp.h>
#define MAX_ITEMS 5000000    // 设置最大元素个数
#define P_ERROR 0.000001     // 设置误差

#define BATCH 16
#define THREADNUM 2
int main() {
    static BaseBloomFilter stBloomFilter = {0};

    std::string graphPath = "/raid/bear/test_dataset/";
    ReadEngine readengine(graphPath);
    std::vector<int64_t> tids;
    readengine.readTrainIdx(tids);
    // for(int i = 0 ; i < tids.size() ; i++) {
    //     std::cout << tids[i]<< std::endl;
    // }
    
    InitBloomFilter(&stBloomFilter, 0, tids.size(), P_ERROR);
    
    if(0 != BloomFilter_AddNodes(&stBloomFilter,tids)){
		std::cout << "add nodes failed " << std::endl;
        exit(0);
	}

    std::pair<int64_t,int64_t> edge(-1,-1);
    
    // std::vector<std::pair<int64_t,int64_t>> ChooseEdges;
    std::vector<int64_t> tmp;
    std::vector<std::vector<int64_t>> ChooseEdgesIdList(THREADNUM,tmp);
    std::vector<std::vector<int64_t>> ChooseNodesList(THREADNUM,tmp);
    std::vector<std::pair<int64_t, int64_t>> edges(BATCH,edge);
    std::vector<int64_t> edgeids(BATCH,0);
    int edgeNUM = BATCH;
    int ret = -1;
    while( -1 != readengine.readlines(edges,edgeids,edgeNUM)) {
        #pragma omp parallel for num_threads(THREADNUM) private(edge,ret)
        for (int i = 0 ;i < BATCH ; i++) {
            int tid = omp_get_thread_num();
            edge = edges[i];
            if (edge.first == -1)   continue;
            //#pragma omp critical
            ret = BloomFilter_CheckEdge(&stBloomFilter,edge);
            std::cout << "thread " << tid << " get edges:" << edge.first << "-->" << edge.second << " || ret --> "<< ret  << std::endl;
            if(ret == 1)
            {
                // std::cout << "ret " << edgeids[i] << "--"<< std::endl;
                ChooseEdgesIdList[tid].emplace_back(edge.second);
                ChooseNodesList[tid].emplace_back(edgeids[i]);
            }
            else if(ret == 2)
            {
                // std::cout << "ret " << edgeids[i] << "--"<< std::endl;
                // std::cout << "thread " << tid << " get edges:" << edge.first << "-->" << edge.second << " || ret --> "<< ret  << std::endl;
                ChooseEdgesIdList[tid].emplace_back(edge.first);
                ChooseNodesList[tid].emplace_back(edgeids[i]);
            }
        }
    }
    int64_t sum = 0;
    for(int i = 0 ; i < THREADNUM ; i++) {
        sum += ChooseEdgesIdList[i].size();
    }
    std::cout << "one hop num :" << sum << std::endl;
    std::cout << "all edges num :" << readengine.readPtr << std::endl;
    return 0;
}