#ifndef PARTITIONER_H
#define PARTITIONER_H
#include <vector>
#include <unordered_map>
#include "StreamCluster.h"
#include "graph.h"
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
    Graph* graph;
    StreamCluster streamCluster_S;
    StreamCluster streamCluster;
    StreamCluster streamCluster_B;
    int gameRoundCnt;
    std::vector<int> partitionLoad;
    std::vector<int> degree;
    std::vector<std::vector<char>> v2p; 
    std::unordered_map<int, int> clusterPartition;
    GlobalConfig config;

    Partitioner();
    Partitioner(StreamCluster streamCluster,GlobalConfig config);
    void performStep();
    int getGameRoundCnt();
    std::unordered_map<int, int> getClusterPartition();
    double getReplicateFactor();
    double getLoadBalance();
    void startStackelbergGame();

private:
    void processGraph(double maxLoad);
};

#endif // PARTITIONER_H
