#ifndef CLUSTER_GAME_TASK_H
#define CLUSTER_GAME_TASK_H

#include <vector>
#include <string>
#include "StreamCluster.h"
#include "ClusterPackGame.h"


class ClusterGameTask {
private:
    StreamCluster streamCluster;
    StreamCluster streamCluster_B;
    StreamCluster streamCluster_S;
    std::vector<int> cluster;
    std::vector<int> cluster_B;
    std::vector<int> cluster_S;
    GlobalConfig config;
    std::string graphType;
public:
    ClusterGameTask(std::string graphType, int taskId, StreamCluster streamCluster,GlobalConfig config);
    ClusterGameTask(std::string graphType, StreamCluster streamCluster, int taskId_B, int taskId_S,GlobalConfig config);
    ClusterPackGame call();
};

#endif // CLUSTER_GAME_TASK_H
