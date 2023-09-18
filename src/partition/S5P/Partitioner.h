#ifndef PARTITIONER_H
#define PARTITIONER_H

#include "StreamCluster.h"
#include "globalConfig.h"
#include "ClusterGameTask.h"
#include "common.h"

// #include "CThreadPool/src/CThreadPool.h"

class Partitioner {
public:
    StreamCluster streamCluster;
    int gameRoundCnt;
    std::vector<int> partitionLoad;
    std::vector<std::vector<bool>> v2p; 
    // phmap::flat_hash_map<int, uint8_t> clusterPartition;
    GlobalConfig config;
    Partitioner();
    Partitioner(StreamCluster streamCluster,GlobalConfig config);
    void performStep();
    std::unordered_map<int, int> getClusterPartition();
    double getReplicateFactor();
    double getLoadBalance();
    void startStackelbergGame();
};

#endif // PARTITIONER_H
