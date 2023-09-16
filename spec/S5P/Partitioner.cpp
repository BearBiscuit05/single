#include "Partitioner.h"
extern std::unordered_map<int, int> clusterPartition;
int __readStep(std::ifstream& fileStream,Edge& edge) {
    std::string line;
    if (std::getline(fileStream, line)) {
        if (line.empty() || line[0] == '#')
            return __readStep(fileStream,edge);

        size_t tabPos = line.find('\t');
        int srcVId = std::stoi(line.substr(0, tabPos));
        int destVId = std::stoi(line.substr(tabPos + 1));

        if (srcVId == destVId)
            return __readStep(fileStream,edge);
        edge.srcVId = srcVId;
        edge.destVId = destVId;
        edge.weight = 1;
        return 0;
    }
    std::cout << "read end..." << std::endl;
    return -1; // Return an empty edge if end of file is reached
}

Partitioner::Partitioner() {}

Partitioner::Partitioner(StreamCluster streamCluster,GlobalConfig config)
    : streamCluster(streamCluster), graph(streamCluster.graph) {
    this->gameRoundCnt = 0;
    this->config = config;
    // std::cout << "partitioner init:" << config.batchSize << std::endl;
    partitionLoad.resize(config.partitionNum);
    degree = streamCluster.getDegree();
    std::cout << graph->getVCount() << std::endl;
    v2p.resize(graph->getVCount(), std::vector<char>(config.partitionNum));
    // std::cout << "partitioner init config.batchSiz :" << config.batchSize << std::endl;
    // std::cout << "partitioner init streamCluster.config.batchSize:" << this->streamCluster.config.batchSize << std::endl;
}

void Partitioner::performStep() {
    double maxLoad = static_cast<double>(config.eCount) / config.partitionNum * 1.1;
    processGraph(maxLoad);
}

void Partitioner::processGraph(double maxLoad) {
    Edge edge(-1,-1,-1);
    std::string inputGraphPath = config.inputGraphPath;
    std::ifstream tmp(inputGraphPath);
    while (-1 != __readStep(tmp,edge)) {
        int src = edge.getSrcVId();
        int dest = edge.getDestVId();
        if (degree[src] >= config.getTao() * config.getAverageDegree() &&
            degree[dest] >= config.getTao() * config.getAverageDegree()) {
            int srcPartition = clusterPartition[streamCluster.getClusterId(src, "B")];
            int destPartition = clusterPartition[streamCluster.getClusterId(dest, "B")];
            int edgePartition = -1;

            if (partitionLoad[srcPartition] > maxLoad && partitionLoad[destPartition] > maxLoad) {
                for (int i = 0; i < config.partitionNum; i++) {
                    if (partitionLoad[i] <= maxLoad) {
                        edgePartition = i;
                        srcPartition = i;
                        destPartition = i;
                        break;
                    }
                }
            } else if (partitionLoad[srcPartition] > partitionLoad[destPartition]) {
                edgePartition = destPartition;
                srcPartition = destPartition;
            } else {
                edgePartition = srcPartition;
                destPartition = srcPartition;
            }
            partitionLoad[edgePartition]++;
            v2p[src][srcPartition] = 1;
            v2p[dest][destPartition] = 1;
        } else {
            int srcPartition = clusterPartition[streamCluster.getClusterId(src, "S")];
            int destPartition = clusterPartition[streamCluster.getClusterId(dest, "S")];
            int edgePartition = -1;
            if (partitionLoad[srcPartition] > maxLoad && partitionLoad[destPartition] > maxLoad) {
                for (int i = config.partitionNum - 1; i >= 0; i--) {
                    if (partitionLoad[i] <= maxLoad) {
                        edgePartition = i;
                        srcPartition = i;
                        destPartition = i;
                        break;
                    }
                }
            } else if (partitionLoad[srcPartition] > partitionLoad[destPartition]) {
                edgePartition = destPartition;
                srcPartition = destPartition;
            } else {
                edgePartition = srcPartition;
                destPartition = srcPartition;
            }
            partitionLoad[edgePartition]++;
            v2p[src][srcPartition] = 1;
            v2p[dest][destPartition] = 1;
        }
    }
}

int Partitioner::getGameRoundCnt() {
    return gameRoundCnt;
}

std::unordered_map<int, int> Partitioner::getClusterPartition() {
    return clusterPartition;
}

double Partitioner::getReplicateFactor() {
    int replicateTotal = 0;
    for (int i = 0; i < graph->getVCount(); i++) {
        for (int j = 0; j < config.partitionNum; j++) {
            replicateTotal += v2p[i][j];
        }
    }
    return static_cast<double>(replicateTotal) / config.vCount;
}

double Partitioner::getLoadBalance() {
    int maxLoad = 0;
    for (int i = 0; i < config.partitionNum; i++) {
        if (maxLoad < partitionLoad[i]) {
            maxLoad = partitionLoad[i];
        }
    }
    return static_cast<double>(config.partitionNum) / config.eCount * maxLoad;
}

void Partitioner::startStackelbergGame() {
    int threads = config.getThreads();
    std::vector<std::thread> threadPool;
    // std::queue<std::future<ClusterPackGame>> futureList;
    std::vector<ClusterPackGame> test_futureList;
    // std::vector<std::unordered_map<int, int>> clusterPartitions_S;
    // std::vector<std::unordered_map<int, int>> clusterPartitions_B;
    int batchSize = config.getBatchSize();
    std::vector<int> clusterList_B = streamCluster.getClusterList_B();
    std::vector<int> clusterList_S = streamCluster.getClusterList_S();
    int clusterSize_B = clusterList_B.size();
    int clusterSize_S = clusterList_S.size();

    // std::cout << clusterSize_B << std::endl;
    // std::cout << clusterSize_S << std::endl;
    int taskNum_B = (clusterSize_B + batchSize - 1) / batchSize;
    int taskNum_S = (clusterSize_S + batchSize - 1) / batchSize;
    int i = 0, j = 0;

    // std::unique_ptr<CTP::UThreadPool> pool(new CTP::UThreadPool());
    // CTP::UThreadPoolPtr tp = pool.get();
    // std::cout << taskNum_B << " " << taskNum_S << std::endl;

    for (; i < taskNum_B && j < taskNum_S; i++, j++) {
        // std::cout << "start hybrid" << std::endl;
         test_futureList.push_back(
            ClusterGameTask("hybrid", streamCluster, i, j,config).call());
        
        // futureList.push(
        //     tp->commit([this, i, j] { return ClusterGameTask("hybrid", streamCluster, i, j,config).call();})
        // );
        
        // futureList.push_back(std::async(std::launch::async, [this, i, j] {
        //     return ClusterGameTask("hybrid", streamCluster, i, j,config).call();
        // }));
    }

    std::cout << "start B... " << taskNum_B << std::endl;

    
    for (; i < taskNum_B; ++i) {

        test_futureList.push_back(
            ClusterGameTask("B", i, streamCluster,config).call());

        // futureList.push(
        //     tp->commit([this, i] { return ClusterGameTask("B", i, streamCluster,config).call();})
        // );

        // futureList.push_back(std::async(std::launch::async, [this, i] {
        //     return ClusterGameTask("B", i, streamCluster,config).call();
        // }));
    }

    std::cout << "start S... "<< taskNum_S  << std::endl;
    for (; j < taskNum_S; ++j) {
        test_futureList.push_back(
            ClusterGameTask("S", j, streamCluster,config).call());
        
        // futureList.push(
        //     tp->commit([this, j] { return ClusterGameTask("S", j, streamCluster,config).call();})
        // );
        
        // futureList.push_back(std::async(std::launch::async, [this, j] {
        //     return ClusterGameTask("S", j, streamCluster,config).call();
        // }));
    }

    std::cout << "max p ..." << std::max(taskNum_B, taskNum_S) << std::endl;
    for (int p = 0; p < std::max(taskNum_B, taskNum_S); p++) {
        try {
            //TODO
            // ClusterPackGame game = futureList.front().get();
            // futureList.pop();
            ClusterPackGame game = test_futureList[p];
            // if (game.getGraphType() == "S") {
            //     //TODO
            //     game.getClusterPartition();
            //     // this->clusterPartition.push_back(game.getClusterPartition());
            // } else if (game.getGraphType() == "B") {
            //     game.getClusterPartition();
            //     // this->clusterPartition.push_back(game.getClusterPartition());
            // } else if (game.getGraphType() == "hybrid") {
            //     game.getClusterPartition();
            //     // this->clusterPartition.push_back(game.getClusterPartition_S());
            //     // this->clusterPartition.push_back(game.getClusterPartition_B());
            // }
            //  std::cout << "game roundCnt:" << game.getRoundCnt() << std::endl;
            gameRoundCnt += game.getRoundCnt();
        } catch (const std::exception& e) {
            
            std::cerr << e.what() << std::endl;
        }
    }

    // Merge cluster partitions
    // std::cout << "Merge cluster partitions" << std::endl;
    // for (auto& clusterPartition_S : clusterPartitions_S) {
    //     for (auto& kvp : clusterPartition_S) {
    //         clusterPartition_S[kvp.first] = kvp.second;
    //     }
    // }

    // for (auto& clusterPartition_B : clusterPartitions_B) {
    //     for (auto& kvp : clusterPartition_B) {
    //         clusterPartition_B[kvp.first] = kvp.second;
    //     }
    // }
}

