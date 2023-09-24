#ifndef PARTITIONER_H
#define PARTITIONER_H


#include "ClusterGameTask.h"

// #include "CThreadPool/src/CThreadPool.h"

class Partitioner {
public:
    StreamCluster* streamCluster;
    int gameRoundCnt_hybrid;
    int gameRoundCnt_inner;
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
    void findPid_B(std::vector<std::pair<int,int>>& ids,std::vector<int>& pids,int& ptr);
    void findPid_S(std::vector<std::pair<int,int>>& ids,std::vector<int>& pids,int& ptr);
    void setPid(std::vector<std::pair<int,int>>& ids,std::vector<int>& pids,int& ptr);
};

#endif // PARTITIONER_H
