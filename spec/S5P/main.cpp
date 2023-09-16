#include "globalConfig.h"
#include "graph.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <thread>
#include <future>
#include <chrono>
#include "StreamCluster.h"
#include "ClusterPackGame.h"
#include "ClusterGameTask.h"
#include "Partitioner.h"
// #include "Partitioner.h"


void printParaInfo(GlobalConfig& configInfo) {
    std::cout << "input graph: " << configInfo.inputGraphPath << std::endl;
    std::cout << "outputGraphPath: " << configInfo.outputGraphPath << std::endl;
    std::cout << "vCount: " << configInfo.vCount << std::endl;
    std::cout << "eCount: " << configInfo.eCount << std::endl;
    std::cout << "averageDegree: " << configInfo.getAverageDegree() << std::endl;
    std::cout << "partitionNum: " << configInfo.partitionNum << std::endl;
    std::cout << "alpha: " << configInfo.alpha << std::endl;
    std::cout << "beta: " << configInfo.beta << std::endl;
    std::cout << "k: " << configInfo.k << std::endl;
    std::cout << "batchSize: " << configInfo.batchSize << std::endl;
    std::cout << "partitionNum: " << configInfo.partitionNum << std::endl;
    std::cout << "threads: " << configInfo.threads << std::endl;
    std::cout << "MaxClusterVolume: " << configInfo.getMaxClusterVolume() << std::endl;
}

class TeeOutputStream : public std::ostream {
public:
    TeeOutputStream(std::ostream& out1, std::ostream& out2) : out1(out1), out2(out2) {}

    template <typename T>
    TeeOutputStream& operator<<(const T& value) {
        out1 << value;
        out2 << value;
        return *this;
    }

private:
    std::ostream& out1;
    std::ostream& out2;
};

using namespace std;
// std::string inputGraphPath = "/home/dzz/graphdataset/LJ/Dcom-lj.ungraph.txt";
std::string inputGraphPath = "/home/dzz/graphdataset/graphpartition_smallgraph/Dmusae/Dmusae_DE.edges";

int main() {
    GlobalConfig configInfo("./project.properties");
    configInfo.inputGraphPath = inputGraphPath;
    Graph graph(configInfo);
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "---------------start-------------" << std::endl;
    printParaInfo(configInfo);
    std::cout << "---------------loading end-------------" << std::endl;
    // //Record start Mem and Time
    int threads = 4;
    std::vector<std::thread> threadPool;
    std::vector<std::future<void>> futureList;


    std::cout << "Start Time" << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();
    auto ClusterStartTime = std::chrono::high_resolution_clock::now();
    StreamCluster streamCluster(graph, configInfo);

    auto InitialClusteringTime = std::chrono::high_resolution_clock::now();
    streamCluster.startStreamCluster();


    auto ClusteringTime = std::chrono::high_resolution_clock::now();

    streamCluster.computeHybridInfo();
    std::cout << "End Clustering" << std::endl;
    std::cout << "partitioner config:" << configInfo.batchSize << std::endl;
    auto ClusterEndTime = std::chrono::high_resolution_clock::now();
    std::cout << "Big clustersize:" << streamCluster.getClusterList_B().size() << std::endl;
    std::cout << "Small clustersize:" << streamCluster.getClusterList_S().size()<< std::endl;

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(ClusterEndTime - ClusteringTime);
    std::cout << "ComputeHybridInfo time: " << duration.count() << " ms" << std::endl;
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(ClusteringTime - InitialClusteringTime);
    std::cout << "Clustering Core time: " << duration.count() << " ms" << std::endl;
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(InitialClusteringTime - ClusterStartTime);
    std::cout << "Initial Clustering time: " << duration.count() << " ms" << std::endl;

    Partitioner partitioner(streamCluster,configInfo);
    std::cout << "partitioner config:" << partitioner.config.batchSize << std::endl;
    auto gameStartTime = std::chrono::high_resolution_clock::now();
    std::cout << "start Game" << std::endl;
    partitioner.startStackelbergGame();
    auto gameEndTime = std::chrono::high_resolution_clock::now();
    std::cout << "End Game" << std::endl;

    duration = std::chrono::duration_cast<std::chrono::milliseconds>(gameEndTime - gameStartTime);
    std::cout << "Cluster game time: " << duration.count() << " ms" << std::endl;

    auto performStepStartTime = std::chrono::high_resolution_clock::now();
    partitioner.performStep();
    auto performStepEndTime = std::chrono::high_resolution_clock::now();

    auto endTime = std::chrono::high_resolution_clock::now();
    std::cout << "End Time" << std::endl;

    double rf = partitioner.getReplicateFactor();
    double lb = partitioner.getLoadBalance();
    graph.clear();
    int roundCnt = partitioner.getGameRoundCnt();

    std::cout << "Partition num:" << configInfo.getPartitionNum() << std::endl;
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "Partition time: " << duration.count() << " ms" << std::endl;
    std::cout << "Relative balance load:" << lb << std::endl;
    std::cout << "Replicate factor: " << rf << std::endl;
    // std::cout << "Memory cost: " << memoryUsed << " MB" << std::endl;
    std::cout << "Total game round:" << roundCnt << std::endl;
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(gameEndTime - gameStartTime);
    std::cout << "Cluster game time: " << duration.count() << " ms" << std::endl;
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(ClusterEndTime - ClusterStartTime);
    std::cout << "Cluster Time: " << duration.count() << " ms" << std::endl;
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(performStepEndTime - performStepStartTime);
    std::cout << "perform Step Time: " << duration.count() << " ms" << std::endl;
    std::cout << "---------------end-------------" << std::endl;

    return 0;
}
