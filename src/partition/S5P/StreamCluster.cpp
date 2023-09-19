#include "StreamCluster.h"
#include <iostream>
#include <algorithm>
#include <fstream>
#include "readGraph.h"




StreamCluster::StreamCluster() {}

StreamCluster::StreamCluster(GlobalConfig& config) 
    : config(config) {
    this->cluster_B.resize(size_t(config.vCount),-1);
    this->cluster_S.resize(size_t(config.vCount),-1);
    this->volume_B.resize(size_t(0.1 * config.vCount),0);
    this->volume_S.resize(size_t(0.1 * config.vCount),0);
    maxVolume = config.getMaxClusterVolume();
    degree.resize(config.vCount,0);
    degree_S.resize(config.vCount,0);
    calculateDegree();
}

void StreamCluster::startStreamCluster() {
    double averageDegree = config.getAverageDegree();
    int clusterID_B = 0;
    int clusterID_S = 0;
    int clusterNUM = config.vCount;
    std::cout << "start read Streaming Clustring..." << std::endl;
    std::string inputGraphPath = config.inputGraphPath;
    std::string line;
    std::pair<int,int> edge(-1,-1);
    this->isInB.resize(config.eCount,false);
    TGEngine tgEngine(inputGraphPath,NODENUM,EDGENUM);  
    while (-1 != tgEngine.readline(edge)) {
        int src = edge.first;
        int dest = edge.second;
        if (degree[src] >= config.tao * averageDegree && degree[dest] >= config.tao * averageDegree) {
            this->isInB[tgEngine.readPtr/2] = true;
            if (cluster_B[src] == -1) {
                cluster_B[src] = clusterID_B++;
            }
            if (cluster_B[dest] == -1) {
                cluster_B[dest] = clusterID_B++;
            }
            if (cluster_B[src] >= volume_B.size() || cluster_B[dest] >= volume_B.size()) {
                volume_B.resize(volume_B.size() + 0.1 * config.vCount, 0);
            }
            volume_B[cluster_B[src]]++;
            volume_B[cluster_B[dest]]++;
            if (volume_B[cluster_B[src]] >= maxVolume) {
                volume_B[cluster_B[src]] -= degree[src];
                cluster_B[src] = clusterID_B++;
                volume_B[cluster_B[src]] = degree[src];
            }
            if (volume_B[cluster_B[dest]] >= maxVolume) {
                volume_B[cluster_B[dest]] -= degree[dest];
                cluster_B[dest] = clusterID_B++;
                volume_B[cluster_B[dest]] = degree[dest];
            }
            
            this->innerAndCutEdge[std::to_string(cluster_B[src]) + "," + std::to_string(cluster_B[dest])] += 1;
            
        } else {
            if (cluster_S[src] == -1) 
                cluster_S[src] = clusterID_S++;
            if (cluster_S[dest] == -1) 
                cluster_S[dest] = clusterID_S++;
            degree_S[src]++;
            degree_S[dest]++;

            if (cluster_S[src] >= volume_S.size() || cluster_S[dest] >= volume_S.size()) 
                volume_S.resize(volume_S.size() + 0.1 * config.vCount, 0);

            volume_S[cluster_S[src] ]++;
            volume_S[cluster_S[dest]]++;

            if (volume_S[cluster_S[src]] >= maxVolume || volume_S[cluster_S[dest]] >= maxVolume)
                continue;

            int minVid = (volume_S[cluster_S[src]] < volume_S[cluster_S[dest]] ? src : dest);
            int maxVid = (src == minVid ? dest : src);

            if ((volume_S[cluster_S[maxVid]] + degree_S[minVid]) <= maxVolume) {
                volume_S[cluster_S[maxVid]] += degree_S[minVid];
                volume_S[cluster_S[minVid]] -= degree_S[minVid];
                cluster_S[minVid] = cluster_S[maxVid];
            }    
            this->innerAndCutEdge[std::to_string(cluster_S[src] + config.vCount) + "," + std::to_string(cluster_S[dest] + config.vCount)] += 1;
            if (cluster_B[src] !=-1) {
                this->innerAndCutEdge[std::to_string(cluster_B[dest]) + "," + std::to_string(cluster_S[src] + config.vCount)] += 1;
            }
            if (cluster_B[dest] != -1) {
                this->innerAndCutEdge[std::to_string(cluster_B[src]) + "," + std::to_string(cluster_S[dest] + config.vCount)] += 1;
            }   
        }
    }
    // std::cout << 666 << std::endl;
    for (int i = 0; i < volume_B.size(); ++i) {
        if (volume_B[i] != 0)
            clusterList_B.push_back(i);
    }
    volume_B.clear();  

    for (int i = 0; i < volume_S.size(); ++i) {
        if (volume_S[i] != 0)
            clusterList_S.push_back(i + config.vCount);
    }
    volume_S.clear();  
    this->config.clusterBSize = config.vCount;
}


void StreamCluster::computeHybridInfo() {
    /*
    std::string inputGraphPath = config.inputGraphPath;
    std::pair<int,int> edge(-1,-1);
    TGEngine tgEngine(inputGraphPath,NODENUM,EDGENUM); 
    int clusterNUM = this->getClusterList_B().size() + this->getClusterList_S().size();
    for(int i = 0 ; i < cluster_S.size() ; i++) {
        cluster_S[i] += cluster_B.size();
    }
    int b_size = cluster_B.size();
    while (-1 != tgEngine.readline(edge)) {
        int src = edge.first;
        int dest = edge.second;
        if (this->isInB[tgEngine.readPtr/2]) {
            this->innerAndCutEdge[cluster_B[src]*clusterNUM + cluster_B[dest]] += 1;
        } else {
            this->innerAndCutEdge[cluster_S[src]*clusterNUM + cluster_S[dest]] += 1;
            if (cluster_B[src] != b_size) {
                this->innerAndCutEdge[cluster_B[dest]*clusterNUM + cluster_S[src]] += 1;
            }
            if (cluster_B[dest] != b_size) {
                this->innerAndCutEdge[cluster_B[src]*clusterNUM + cluster_S[dest]] += 1;
            }
        } 
    }
    */
}

void StreamCluster::calculateDegree() {
    std::pair<int,int> edge(-1,-1);
    std::string inputGraphPath = config.inputGraphPath;
    TGEngine tgEngine(inputGraphPath,NODENUM,EDGENUM);  
    // std::cout << "count :"  << count << std::endl;
    while (-1 != tgEngine.readline(edge)) {
        int src = edge.first;
        int dest = edge.second;
        degree[src] ++;
        degree[dest] ++;
    }
    std::cout << "End CalculateDegree" << std::endl;
}

int StreamCluster::getEdgeNum(int cluster1, int cluster2) {
    std::string index = std::to_string(cluster1) + "," + std::to_string(cluster2);
    if(innerAndCutEdge.find(index) != innerAndCutEdge.end()) {
        return innerAndCutEdge[index];
    }
    return 0;
}

std::vector<int> StreamCluster::getClusterList_B() {
    return clusterList_B;
}

std::vector<int> StreamCluster::getClusterList_S() {
    return clusterList_S;
}













