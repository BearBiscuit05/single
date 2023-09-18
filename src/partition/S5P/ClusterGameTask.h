#ifndef CLUSTER_GAME_TASK_H
#define CLUSTER_GAME_TASK_H

#include "common.h"
#include "StreamCluster.h"


class ClusterGameTask {
public:
    StreamCluster* streamCluster;
    std::vector<int> cluster;
    std::vector<int> cluster_B;
    std::vector<int> cluster_S;
    std::string graphType;
    GlobalConfig* config;
    int roundCnt = 0;
    double beta = 0.0;
    double beta_B = 0.0;
    double beta_S = 0.0;
    std::vector<double> partitionLoad;
    std::unordered_map<int, int> cutCostValue;
    std::unordered_map<int, std::unordered_set<int>> clusterNeighbours;


    ClusterGameTask(std::string graphType, int taskId, StreamCluster& streamCluster);
    ClusterGameTask(std::string graphType, StreamCluster& streamCluster, int taskIds);
    void call();
    void startGameDouble();
    void initGame();
    double computeCost(int clusterId, int partition, const std::string type);
};

#endif // CLUSTER_GAME_TASK_H
