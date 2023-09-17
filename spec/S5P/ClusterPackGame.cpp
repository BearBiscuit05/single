#include "ClusterPackGame.h"
std::unordered_map<int, int> clusterPartition = std::unordered_map<int, int>();

ClusterPackGame::ClusterPackGame() {}

ClusterPackGame::ClusterPackGame(StreamCluster streamCluster, std::vector<int>& clusterList,std::string& graphType,GlobalConfig& config) {
    this->config = config;
    this->graphType = graphType;
    this->streamCluster = streamCluster;
    this->clusterList = clusterList;
    this->partitionLoad.resize(config.partitionNum,0);

}

ClusterPackGame::ClusterPackGame(StreamCluster streamCluster, std::vector<int>& clusterList_B, std::vector<int>& clusterList_S,std::string& graphType,GlobalConfig& config) {
    this->config = config;
    this->streamCluster = streamCluster;
    this->cutCostValue = std::unordered_map<int, int>();
    this->partitionLoad.resize(config.partitionNum,0);
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
        double minLoad = config.eCount;
        for (int i = 0; i < config.partitionNum; i++) {
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
    int partition = 0;
    for (int clusterId : clusterList_B) {
        double minLoad = config.eCount;
        for (int i = 0; i < config.partitionNum; i++) {
            if (partitionLoad[i] < minLoad) {
                minLoad = partitionLoad[i];
                partition = i;
            }
        }
        clusterPartition[clusterId] = partition;
        partitionLoad[partition] += streamCluster.getEdgeNum(clusterId, clusterId);
    }

    for (int clusterId : clusterList_S) {
        double minLoad = config.eCount;
        for (int i = 0; i < config.partitionNum; i++) {
            if (partitionLoad[i] < minLoad) {
                minLoad = partitionLoad[i];
                partition = i;
            }
        }
        clusterPartition[clusterId] = partition;
        partitionLoad[partition] += streamCluster.getEdgeNum(clusterId, clusterId);
    }

    double sizePart_B = 0.0, cutPart_B = 0.0;
    double sizePart_S = 0.0, cutPart_S = 0.0;

    for (int cluster1 : clusterList_B) {
        sizePart_B += streamCluster.getEdgeNum(cluster1, cluster1);
        for (int cluster2 : clusterList_B) {
            int innerCut = 0;
            if (cluster1 != cluster2) {
                innerCut = streamCluster.getEdgeNum(cluster1, cluster2);
                if (innerCut != 0) {
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

        for (int cluster2 : clusterList_S) {
            int innerCut = 0;
            if (cluster1 != cluster2) {
                innerCut = streamCluster.getEdgeNum(cluster1, cluster2);
                if (innerCut != 0) {
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
            if (cluster1 != cluster2) {
                innerCut = streamCluster.getEdgeNum(cluster1, cluster2);
                if (innerCut != 0) {
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
            if (cluster1 != cluster2) {
                innerCut = streamCluster.getEdgeNum(cluster2,  cluster1);
                if (innerCut != 0) {
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
}


double ClusterPackGame::computeCost(int clusterId, int partition, const std::string type) {
    if (type == "B") {
        double loadPart = 0.0;
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

        double alpha = config.alpha, k = config.partitionNum;
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

        double alpha = config.alpha, k = config.partitionNum;
        double m = streamCluster.getEdgeNum(clusterId, clusterId);


        double Cost = beta_S / k * loadPart * m + edgeCutPart  +  m;
        return Cost;
    } else {
        std::cout << "ComputeCost Error!" << std::endl;
        return 0.0;
    }
}

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
            for (int j = 0; j < config.partitionNum / 2; j++) {
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
            for (int j = config.partitionNum - 1; j >= config.partitionNum / 2; j--) {
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

















