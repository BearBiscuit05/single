#ifndef PARTITIONER_H
#define PARTITIONER_H
#include <vector>
#include <unordered_map>
#include "StreamCluster.h"
#include "globalConfig.h"
#include "ClusterGameTask.h"
#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <thread>
#include <future>
// #include "CThreadPool/src/CThreadPool.h"

class Partitioner {
public:
    StreamCluster streamCluster;
    int gameRoundCnt;
    std::vector<int> partitionLoad;
    std::vector<std::vector<bool>> v2p; 
    std::unordered_map<int, int> clusterPartition;
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
