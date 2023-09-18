#include "Partitioner.h"
#include "readGraph.h"

extern std::unordered_map<int, int> clusterPartition;

Partitioner::Partitioner() {}

Partitioner::Partitioner(StreamCluster streamCluster, GlobalConfig config)
    : streamCluster(streamCluster), config(config) {
    this->gameRoundCnt = 0;
    partitionLoad.resize(config.partitionNum);
    std::cout << config.vCount << std::endl;

    v2p.resize(config.vCount, std::vector<bool>(config.partitionNum));
    
   }

void Partitioner::performStep() {
    double maxLoad = static_cast<double>(config.eCount) / config.partitionNum * 1.1;
    std::string inputGraphPath = config.inputGraphPath;
    std::pair<int,int> edge(-1,-1);
    TGEngine tgEngine(inputGraphPath,3072441,10308445);  
    while (-1 != tgEngine.readline(edge)) {
        int src = edge.first;
        int dest = edge.second;
        if (this->streamCluster.isInB[tgEngine.readPtr/2]) {
            int srcPartition = clusterPartition[streamCluster.cluster_B[src]];
            int destPartition = clusterPartition[streamCluster.cluster_B[dest]];
            int edgePartition = -1;
            if (partitionLoad[srcPartition] > maxLoad && partitionLoad[destPartition] > maxLoad) {
                for (int i = 0; i < config.partitionNum; i++) {
                    if (partitionLoad[i] <= maxLoad) {
                        edgePartition = i;
                        srcPartition = i;
                        destPartition = i;
                        break;
                    }
                }
            } else if (partitionLoad[srcPartition] > partitionLoad[destPartition]) {
                edgePartition = destPartition;
                srcPartition = destPartition;
            } else {
                edgePartition = srcPartition;
                destPartition = srcPartition;
            }
            partitionLoad[edgePartition]++;
            v2p[src][srcPartition] = 1;
            v2p[dest][destPartition] = 1;
        } else {
            int srcPartition = clusterPartition[streamCluster.cluster_S[src]];
            int destPartition = clusterPartition[streamCluster.cluster_S[dest]];
            int edgePartition = -1;
            if (partitionLoad[srcPartition] > maxLoad && partitionLoad[destPartition] > maxLoad) {
                for (int i = config.partitionNum - 1; i >= 0; i--) {
                    if (partitionLoad[i] <= maxLoad) {
                        edgePartition = i;
                        srcPartition = i;
                        destPartition = i;
                        break;
                    }
                }
            } else if (partitionLoad[srcPartition] > partitionLoad[destPartition]) {
                edgePartition = destPartition;
                srcPartition = destPartition;
            } else {
                edgePartition = srcPartition;
                destPartition = srcPartition;
            }
            partitionLoad[edgePartition]++;
            
            // if(src >= config.vCount || dest >= config.vCount || srcPartition >= 32 || destPartition >= 32)
            //     std::cout<< "!!!"<<std::endl;
            v2p[src][srcPartition] = 1;
            v2p[dest][destPartition] = 1;

            
        }
    }
    std::cout << "11" << std::endl; 
}

double Partitioner::getReplicateFactor() {
    int replicateTotal = 0;
    for (int i = 0; i < config.vCount; i++) {
        for (int j = 0; j < config.partitionNum; j++) {
            replicateTotal += v2p[i][j];
        }
    }
    return static_cast<double>(replicateTotal) / config.vCount;
}

double Partitioner::getLoadBalance() {
    int maxLoad = 0;
    for (int i = 0; i < config.partitionNum; i++) {
        if (maxLoad < partitionLoad[i]) {
            maxLoad = partitionLoad[i];
        }
    }
    return static_cast<double>(config.partitionNum) / config.eCount * maxLoad;
}

void Partitioner::startStackelbergGame() {
    //int threads = config.batchSize;
    //std::vector<std::thread> threadPool;
    int batchSize = config.batchSize;
    std::vector<int> clusterList_B = streamCluster.getClusterList_B();
    std::vector<int> clusterList_S = streamCluster.getClusterList_S();
    int taskNum_B = (clusterList_B.size() + batchSize - 1) / batchSize;
    int taskNum_S = (clusterList_S.size() + batchSize - 1) / batchSize;
    int minTaskNUM = std::min(taskNum_B,taskNum_S);
    int leftTaskNUM = std::abs(taskNum_B - taskNum_S);

    omp_set_num_threads(THREADNUM);
    std::cout << taskNum_B << " " << taskNum_S << std::endl;
    
#pragma omp parallel for
    for (int i = 0; i < minTaskNUM; i++) {
        ClusterGameTask("hybrid", streamCluster, i).call();
    }

    std::cout << "start B... " << taskNum_B << std::endl;

 
    if (taskNum_B > taskNum_S) {
#pragma omp parallel for  
        for (int i = 0 ; i < leftTaskNUM; ++i) {
            ClusterGameTask("B", i+minTaskNUM, streamCluster).call();
        }
    } else {
#pragma omp parallel for  
        for (int i = 0 ; i < leftTaskNUM; ++i) {
            ClusterGameTask("S", i+minTaskNUM, streamCluster).call();
        }
    }
    

    std::cout << "start S... "<< taskNum_S  << std::endl;

//#pragma omp parallel for
    

    // std::cout << "max p ..." << std::max(taskNum_B, taskNum_S) << std::endl;
    // for (int p = 0; p < std::max(taskNum_B, taskNum_S); p++) {
    //     try {
    //         // ClusterPackGame game = test_futureList[p];
    //         // gameRoundCnt += game.getRoundCnt();
    //     } catch (const std::exception& e) {
            
    //         std::cerr << e.what() << std::endl;
    //     }
    // }

}

