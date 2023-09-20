#ifndef PARTITIONER_H
#define PARTITIONER_H


#include "ClusterGameTask.h"

// #include "CThreadPool/src/CThreadPool.h"

class Partitioner {
public:
    StreamCluster* streamCluster;
    int gameRoundCnt;
    std::vector<int> partitionLoad;
    std::vector<std::vector<char>> v2p; 
    std::vector<int> clusterPartition;
    GlobalConfig config;
    Partitioner();
    Partitioner(StreamCluster& streamCluster,GlobalConfig config);
    void performStep();
    double getReplicateFactor();
    double getLoadBalance();
    void startStackelbergGame();
};

#endif // PARTITIONER_H
