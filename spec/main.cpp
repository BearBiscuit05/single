#include "readGraph.h"
#include <omp.h>
#define MAX_ITEMS 5000000    // 设置最大元素个数
#define P_ERROR 0.000001     // 设置误差

#define BATCH 1024
#define THREADNUM 8
int main() {
    std::string graphPath = "/raid/bear/test_dataset/";
    ReadEngine readengine(graphPath);
    std::vector<int64_t> tids;
    readengine.readTrainIdx(tids);
    std::pair<int64_t,int64_t> edge(-1,-1);

    std::vector<int> ChooseEdgesIdList(THREADNUM,0);
    std::vector<std::pair<int64_t, int64_t>> edges(BATCH,edge);
    std::vector<int64_t> edgeids(BATCH,0);
    int edgeNUM = BATCH;
    int ret = -1;
    while( -1 != readengine.readlines(edges,edgeids,edgeNUM)) {
        #pragma omp parallel for num_threads(THREADNUM) private(edge,ret)
        for (int i = 0 ;i < BATCH ; i++) {
            int tid = omp_get_thread_num();
            edge = edges[i];           
        }
    }

    int sum = 0;
    for (int element : ChooseEdgesIdList) {
        sum += element;
    }
    std::cout << "one hop num :" << sum << std::endl;
    std::cout << "all edges num :" << readengine.readPtr << std::endl;
    return 0;
}