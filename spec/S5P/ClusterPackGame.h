#pragma once

#include "StreamCluster.h"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <list>

class ClusterPackGame {
private:
    struct PairHash {
        template <class T1, class T2>
        std::size_t operator () (const std::pair<T1, T2>& p) const {
            auto h1 = std::hash<T1>{}(p.first);
            auto h2 = std::hash<T2>{}(p.second);
            // 可以使用一个位运算混合哈希值
            return h1 ^ h2;
        }
    };

    std::unordered_map<int, int> cutCostValue; // key: cluster value: cutCost
    std::unordered_map<int, std::unordered_set<int>> clusterNeighbours;
    std::vector<double> partitionLoad;
    std::vector<int> clusterList;
    StreamCluster streamCluster;
    StreamCluster streamCluster_B;
    StreamCluster streamCluster_S;
    std::vector<int> clusterList_S;
    std::vector<int> clusterList_B;
    double beta = 0.0;
    double beta_B = 0.0;
    double beta_S = 0.0;
    int roundCnt;
    std::string graphType;
    int gap = 0;
    GlobalConfig config;
    
public:
    
    ClusterPackGame();
    ClusterPackGame(StreamCluster streamCluster, std::vector<int>& clusterList, std::string& graphType,GlobalConfig& config);
    ClusterPackGame(StreamCluster streamCluster, std::vector<int>& clusterList_B, std::vector<int>& clusterList_S, std::string& graphType,GlobalConfig& config); 
    ClusterPackGame(StreamCluster streamCluster, std::vector<int>& clusterList,GlobalConfig& config);
    ClusterPackGame(StreamCluster streamCluster_B, std::vector<int>& clusterList_B, StreamCluster& streamCluster_S, std::vector<int>& clusterList_S,GlobalConfig& config);
    std::string getGraphType();
    
    void initGame();
    void initGameDouble();
    double computeCost(int clusterId, int partition);
    double computeCost(int clusterId, int partition, std::string type);
    void startGame();
    void startGameDouble();

    int getRoundCnt();

};