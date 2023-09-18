#ifndef STREAM_CLUSTER_H
#define STREAM_CLUSTER_H

#include "globalConfig.h"
#include "common.h"
class StreamCluster {
public:
    std::vector<int> cluster_S;     
    std::vector<int> cluster_B;     
    std::vector<int> degree;
    std::vector<int> degree_S;
    std::vector<int> volume_S;     
    std::vector<int> volume_B;     
    std::unordered_map<std::string , int> innerAndCutEdge;
    std::vector<int> clusterList_S;
    std::vector<int> clusterList_B;
    int maxVolume;
    int maxVolume_B;
    int maxVolume_S;
    std::string graphType;
    GlobalConfig config;
    StreamCluster();
    StreamCluster(GlobalConfig& config);
    void startStreamCluster();
    void computeHybridInfo();
    void calculateDegree();
    int getEdgeNum(int cluster1, int cluster2);
    std::vector<bool> isInB;
    std::vector<int> getClusterList_B();
    std::vector<int> getClusterList_S();
};
#endif // STREAM_CLUSTER_H
