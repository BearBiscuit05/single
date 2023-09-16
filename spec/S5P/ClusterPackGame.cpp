#include "ClusterPackGame.h"
extern std::unordered_map<int, int> clusterPartition = std::unordered_map<int, int>();

ClusterPackGame::ClusterPackGame() {}

ClusterPackGame::ClusterPackGame(StreamCluster streamCluster, std::vector<int>& clusterList,std::string& graphType,GlobalConfig& config) {
    this->config = config;
    this->graphType = graphType;

    this->streamCluster = streamCluster;
    this->clusterList = clusterList;

    this->partitionLoad.resize(config.getPartitionNum(),0);

}

ClusterPackGame::ClusterPackGame(StreamCluster streamCluster, std::vector<int>& clusterList_B, std::vector<int>& clusterList_S,std::string& graphType,GlobalConfig& config) {
    this->config = config;
    this->streamCluster = streamCluster;
    //TODO
    
    // clusterPartition_B = std::unordered_map<int, int>();
    // clusterPartition_S = std::unordered_map<int, int>();
    clusterList = streamCluster.getClusterList();

    this->cutCostValue = std::unordered_map<int, int>();
    this->partitionLoad.resize(config.getPartitionNum(),0);
    clusterNeighbours = std::unordered_map<int, std::unordered_set<int>>();
    this->clusterList_B = clusterList_B;
    this->clusterList_S = clusterList_S;
    this->graphType = graphType;
    this->roundCnt = 0;
}

std::string ClusterPackGame::getGraphType() {
    return graphType;
}



void ClusterPackGame::initGame() {
    int partition = 0;
    for (int clusterId : clusterList) {
        double minLoad = config.getECount();
        for (int i = 0; i < config.getPartitionNum(); i++) {
            if (partitionLoad[i] < minLoad) {
                minLoad = partitionLoad[i];
                partition = i;
            }
        }
        clusterPartition[clusterId] = partition;
        partitionLoad[partition] += streamCluster.getEdgeNum(clusterId, clusterId);

    }


}

void ClusterPackGame::initGameDouble() {
    // std::cout << "000" << std::endl;
    int partition = 0;
    for (int clusterId : clusterList_B) {
        double minLoad = config.getECount();
        for (int i = 0; i < config.getPartitionNum(); i++) {
            if (partitionLoad[i] < minLoad) {
                minLoad = partitionLoad[i];
                partition = i;
            }
        }
        clusterPartition[clusterId] = partition;
        partitionLoad[partition] += streamCluster.getEdgeNum(clusterId, clusterId);
    }
    // std::cout << "111" << std::endl;
    for (int clusterId : clusterList_S) {
        double minLoad = config.getECount();
        for (int i = 0; i < config.getPartitionNum(); i++) {
            if (partitionLoad[i] < minLoad) {
                minLoad = partitionLoad[i];
                partition = i;
            }
        }
        clusterPartition[clusterId] = partition;
        partitionLoad[partition] += streamCluster.getEdgeNum(clusterId, clusterId);
    }
    // std::cout << "222" << std::endl;
    double sizePart_B = 0.0, cutPart_B = 0.0;
    double sizePart_S = 0.0, cutPart_S = 0.0;

    // std::cout << clusterList_B.size() << std::endl;
    for (int cluster1 : clusterList_B) {
        // std::cout << "111" << std::endl;
        sizePart_B += streamCluster.getEdgeNum(cluster1, cluster1);
        // std::cout << "222" << std::endl;
        for (int cluster2 : clusterList_B) {
            int innerCut = 0;
            int outerCut = 0;
            if (cluster1 != cluster2) {
                innerCut = streamCluster.getEdgeNum(cluster1, cluster2);
                // outerCut = streamCluster.getEdgeNum(cluster1, cluster2, "B");

                if (innerCut != 0 || outerCut != 0) {
                    auto it = clusterNeighbours.find(cluster1);
                    if (it == clusterNeighbours.end())
                        clusterNeighbours[cluster1] = std::unordered_set<int>();
                    clusterNeighbours[cluster1].insert(cluster2);
                }
                cutPart_B += innerCut;
            }
            auto it = cutCostValue.find(cluster1);
            if (it == cutCostValue.end())
                cutCostValue[cluster1] = 0;
            cutCostValue[cluster1] += innerCut;
        }
        // std::cout << "333" << std::endl;
        for (int cluster2 : clusterList_S) {
            int innerCut = 0;
            int outerCut = 0;
            if (cluster1 != cluster2) {
                innerCut = streamCluster.getEdgeNum(cluster1, cluster2);
                // outerCut = streamCluster.getEdgeNum(cluster1, cluster2, "hybrid");
                if (innerCut != 0 || outerCut != 0) {
                    if(clusterNeighbours.find(cluster1) == clusterNeighbours.end()) {
                        clusterNeighbours[cluster1] = std::unordered_set<int>();
                    }
                    clusterNeighbours[cluster1].insert(cluster2);
                }
            }
            if(cutCostValue.find(cluster1) == cutCostValue.end()) {
                cutCostValue[cluster1]  = 0;
            }
            cutCostValue[cluster1] += innerCut;
        }
    }

    beta_B = (double)config.eCount  / (sizePart_B * sizePart_B + 1.0) * (double)config.eCount * ((double)cutPart_B + (double)config.vCount);

    for (int cluster1 : clusterList_S) {
        sizePart_S += streamCluster.getEdgeNum(cluster1, cluster1);
        for (int cluster2 : clusterList_S) {
            int innerCut = 0;
            int outerCut = 0;
            if (cluster1 != cluster2) {
                innerCut = streamCluster.getEdgeNum(cluster1, cluster2);
                // outerCut = streamCluster.getEdgeNum(cluster1, cluster2, "S");
                if (innerCut != 0 || outerCut != 0) {
                    if(clusterNeighbours.find(cluster1) == clusterNeighbours.end()) {
                        clusterNeighbours[cluster1] =std::unordered_set<int>();
                    }
                    clusterNeighbours[cluster1].insert(cluster2);
                }
            }
            
            cutPart_S += innerCut;
            auto it = cutCostValue.find(cluster1);
            if (it == cutCostValue.end())
                cutCostValue[cluster1 ] = 0;
            cutCostValue[cluster1] += innerCut;
        }

        for (int cluster2 : clusterList_B) {
            int innerCut = 0;
            int outerCut = 0;
            if (cluster1 != cluster2) {
                innerCut = streamCluster.getEdgeNum(cluster2,  cluster1);
                // outerCut = streamCluster.getEdgeNum(cluster1, cluster2, "B");

                if (innerCut != 0 || outerCut != 0) {
                    auto it = clusterNeighbours.find(cluster1 );
                    if (it == clusterNeighbours.end())
                        clusterNeighbours[cluster1] = std::unordered_set<int>();
                    clusterNeighbours[cluster1].insert(cluster2);
                }
            }
            if(cutCostValue.find(cluster1) == cutCostValue.end()) {
                cutCostValue[cluster1]  = 0;
            }
            cutCostValue[cluster1] += innerCut;
        }
    }
    beta_S = (double)config.eCount  / (sizePart_S * sizePart_S + 1.0) * (double)config.eCount * ((double)cutPart_S + (double)config.vCount);
    // std::cout << 6666 << std::endl;
}

// double ClusterPackGame::computeCost(int clusterId, int partition) {
//     double loadPart = 0.0;

//     int edgeCutPart = cutCostValue[clusterId];
//     int old_partition = clusterPartition[clusterId];
//     loadPart = partitionLoad[old_partition];
//     if (partition != old_partition)
//         loadPart = partitionLoad[partition] + streamCluster.getEdgeNum(clusterId, clusterId, graphType);
//     auto it = clusterNeighbours.find(clusterId);
//     if (it != clusterNeighbours.end()) {
//         for (int neighbour : clusterNeighbours[clusterId]) {
//             if (clusterPartition[neighbour] == partition)
//                 edgeCutPart = edgeCutPart - streamCluster.getEdgeNum(clusterId, neighbour, graphType)
//                         - streamCluster.getEdgeNum(neighbour, clusterId, graphType);
//         }
//     }

//     double alpha = config.getAlpha(), k = config.getPartitionNum();
//     double m = streamCluster.getEdgeNum(clusterId, clusterId, graphType);

//     return alpha * beta / k * loadPart * m + (1 - alpha) / 2 * edgeCutPart;
// }

double ClusterPackGame::computeCost(int clusterId, int partition, const std::string type) {
    if (type == "B") {
        double loadPart = 0.0;
        // int edgeCutPart_B = cutCostValue_B[clusterId];
        // int edgeCutPart_hybrid_B = cutCostValue_hybrid_B[clusterId];
        double edgeCutPart = cutCostValue[clusterId];
        int old_partition = clusterPartition[clusterId];
        loadPart = partitionLoad[old_partition];
        if (partition != old_partition)
            loadPart = partitionLoad[partition] + streamCluster.getEdgeNum(clusterId, clusterId);


 
        auto it3 = clusterNeighbours.find(clusterId);
        if ( it3 != clusterNeighbours.end()) {
            for (int neighbour : clusterNeighbours[clusterId]) {

                edgeCutPart -= streamCluster.getEdgeNum(clusterId, neighbour);
            }
        }

        double alpha = config.getAlpha(), k = config.getPartitionNum();
        double m = streamCluster.getEdgeNum(clusterId, clusterId);
        double Cost = beta_B / k * loadPart * m +  edgeCutPart   + m;
        return Cost;
    } else if (type == "S") {
        double loadPart = 0.0;
        double edgeCutPart =  cutCostValue[clusterId];
        int old_partition = clusterPartition[clusterId];
        loadPart = partitionLoad[old_partition];
        if (partition != old_partition)
            loadPart = partitionLoad[partition] + streamCluster.getEdgeNum(clusterId, clusterId);

        auto it2 = clusterNeighbours.find(clusterId);
        if ( it2 != clusterNeighbours.end()) {
            for (int neighbour : clusterNeighbours[clusterId]) {
                edgeCutPart -= streamCluster.getEdgeNum(clusterId, neighbour);
            }
        }

        double alpha = config.getAlpha(), k = config.getPartitionNum();
        double m = streamCluster.getEdgeNum(clusterId, clusterId);


        double Cost = beta_S / k * loadPart * m + edgeCutPart  +  m;
        return Cost;
    } else {
        std::cout << "ComputeCost Error!" << std::endl;
        return 0.0;
    }
}

// void ClusterPackGame::startGame() {
//     bool finish = false;
//     while (!finish) {
//         finish = true;
//         for (int clusterId : clusterList) {
//             double minCost = std::numeric_limits<double>::max();
//             int minPartition = clusterPartition[clusterId];

//             if (graphType == "B") {
//                 for (int j = 0; j < config.getPartitionNum() / 2; j++) {
//                     double cost = computeCost(clusterId, j, "B");
//                     if (cost <= minCost) {
//                         minCost = cost;
//                         minPartition = j;
//                     }
//                 }
//             } else if (graphType == "S") {
//                 for (int j = config.getPartitionNum() - 1; j >= config.getPartitionNum() / 2; j--) {
//                     double cost = computeCost(clusterId, j, "S");
//                     if (cost <= minCost) {
//                         minCost = cost;
//                         minPartition = j;
//                     }
//                 }
//             }

//             if (minPartition != clusterPartition[clusterId]) {
//                 finish = false;
//                 partitionLoad[minPartition] += streamCluster.getEdgeNum(clusterId, clusterId, graphType);
//                 partitionLoad[clusterPartition[clusterId]] -= streamCluster.getEdgeNum(clusterId, clusterId, graphType);
//                 clusterPartition[clusterId] = minPartition;
//             }
//         }
//         roundCnt++;
//     }
// }

void ClusterPackGame::startGameDouble() {
    bool finish_B = false;
    bool finish_S = false;
    bool isChangeB = true;
    bool isChangeS = true;
    while (true) {
        finish_B = true;
        finish_S = true;
        for (int clusterId : clusterList_B) {
            double minCost = std::numeric_limits<double>::max();
            int minPartition = clusterPartition[clusterId];
            for (int j = 0; j < config.getPartitionNum() / 2; j++) {
                double cost = computeCost(clusterId, j, "B");
                if (cost <= minCost) {
                    minCost = cost;
                    minPartition = j;
                }
            }

            if (minPartition != clusterPartition[clusterId]) {
                finish_B = false;
                // update partition load
                partitionLoad[minPartition] += streamCluster_B.getEdgeNum(clusterId, clusterId);
                partitionLoad[clusterPartition[clusterId]] -= streamCluster_B.getEdgeNum(clusterId, clusterId, "B");
                clusterPartition[clusterId] = minPartition;
            }
        }

        for (int clusterId : clusterList_S) {
            double minCost = std::numeric_limits<double>::max();
            int minPartition = clusterPartition[clusterId];
            for (int j = config.getPartitionNum() - 1; j >= config.getPartitionNum() / 2; j--) {
                double cost = computeCost(clusterId, j, "S");
                if (cost <= minCost) {
                    minCost = cost;
                    minPartition = j;
                }
            }

            if (minPartition != clusterPartition[clusterId]) {
                finish_S = false;
                // update partition load
                partitionLoad[minPartition] += streamCluster_S.getEdgeNum(clusterId, clusterId);
                partitionLoad[clusterPartition[clusterId]] -= streamCluster_S.getEdgeNum(clusterId, clusterId, "S");
                clusterPartition[clusterId] = minPartition;
            }
        }
        roundCnt++;
        if (finish_B && finish_S) {
            break;
        }
        // std::cout << roundCnt << std::endl;
        // break;
    }
}

int ClusterPackGame::getRoundCnt() {
    return roundCnt;
}

// std::unordered_map<int, int> ClusterPackGame::getClusterPartition() {
//     return clusterPartition;
// }

// std::unordered_map<int, int> ClusterPackGame::getClusterPartition_B() {
//     return clusterPartition_B;
// }

// std::unordered_map<int, int> ClusterPackGame::getClusterPartition_S() {
//     return clusterPartition_S;
// }

















