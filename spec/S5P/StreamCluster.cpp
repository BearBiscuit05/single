#include "StreamCluster.h"
#include <iostream>
#include <algorithm>
#include <fstream>
#include "readGraph.h"




StreamCluster::StreamCluster() {}

StreamCluster::StreamCluster(Graph& graph, GlobalConfig& config) {
    this->config = config;
    this->cluster_B.resize(size_t(config.vCount),-1);
    this->cluster_S.resize(size_t(config.vCount),-1);

    this->volume_B.resize(size_t(0.1 * config.vCount),0);
    this->volume_S.resize(size_t(0.1 * config.vCount),0);
    this->graph = &graph;
    maxVolume = config.getMaxClusterVolume();
    degree.resize(config.vCount,0);
    degree_S.resize(config.vCount,0);
    calculateDegree();
}


void StreamCluster::setDegree(std::vector<int> degree) {
    this->degree = degree;
}



void StreamCluster::setMaxVolume(int maxVolume) {
    this->maxVolume = maxVolume;
}

void StreamCluster::setInnerAndCutEdge(std::unordered_map<int, std::unordered_map<int, int>> innerAndCutEdge) {
    innerAndCutEdge = innerAndCutEdge;
}

void StreamCluster::startStreamCluster() {
    double averageDegree = config.getAverageDegree();
    int clusterID_B = 0;
    int clusterID_S = 0;
    std::cout << "start read Streaming Clustring..." << std::endl;
    std::string inputGraphPath = config.inputGraphPath;
    std::string line;
    std::pair<int,int> edge(-1,-1);
    TGEngine tgEngine(inputGraphPath,3997962,16539643);  
    while (-1 != tgEngine.readline(edge)) {
        int src = edge.first;
        int dest = edge.second;
        if (degree[src] >= config.tao * averageDegree && degree[dest] >= config.tao * averageDegree) {
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
        } else {
            if (cluster_S[src] == -1) {
                cluster_S[src] = clusterID_S++;
            }
            if (cluster_S[dest] == -1) {
                cluster_S[dest] = clusterID_S++;
            }
            degree_S[src]++;
            degree_S[dest]++;

            
            if (cluster_S[src] >= volume_S.size() || cluster_S[dest] >= volume_S.size()) {
                volume_S.resize(volume_S.size() + 0.1 * config.vCount, 0);
            }
            // update volume

            volume_S[cluster_S[src]]++;
            volume_S[cluster_S[dest]]++;

            if (volume_S[cluster_S[src]] >= maxVolume || volume_S[cluster_S[dest]] >= maxVolume)
                continue;

            int minVid = (volume_S[cluster_S[src]] < volume_S[cluster_S[dest]] ? src : dest);
            int maxVid = (src == minVid ? dest : src);

            if ((volume_S[cluster_S[maxVid]] + degree_S[minVid]) <= maxVolume) {
                volume_S[cluster_S[maxVid]] += degree_S[minVid];
                volume_S[cluster_S[minVid]] -= degree_S[minVid];
                // if (volume_S[cluster_S[minVid]] == 0)
                //     volume_S.erase(cluster_S[minVid]);
                cluster_S[minVid] = cluster_S[maxVid];
            }           
        }
    }
    
    //TODO
    // std::vector<std::pair<int, int>> sortList_B(volume_B.begin(), volume_B.end());

    for (int i = 0; i < volume_B.size(); ++i) {
        // if (volume_B[i] == 0) {
        //     continue;
        // }
        clusterList_B.push_back(i);
    }
    volume_B.clear();  



    
    for (int i = 0; i < volume_S.size(); ++i) {

        clusterList_S.push_back(i + cluster_B.size());
    }
    volume_S.clear();  
    this->config.clusterBSize = cluster_B.size();
}

void StreamCluster::computeHybridInfo() {
    std::string inputGraphPath = config.inputGraphPath;
    std::ifstream tmp(inputGraphPath);
    std::pair<int,int> edge(-1,-1);
    TGEngine tgEngine(inputGraphPath,3997962,16539643);  
    while (-1 != tgEngine.readline(edge)) {
        int src = edge.first;
        int dest = edge.second;
        int oldValue = 0;
        if (degree[src] >= config.tao * config.getAverageDegree() && degree[dest] >= config.tao * config.getAverageDegree()) {
            this->innerAndCutEdge[std::make_pair(cluster_B[src], cluster_B[dest])] += 1;
        } else {
            this->innerAndCutEdge[std::make_pair(cluster_S[src] + cluster_B.size(), cluster_S[dest] + cluster_B.size())] += 1;
            if (cluster_B[src] != 0) {
                this->innerAndCutEdge[std::make_pair(cluster_B[dest], cluster_S[src] + cluster_B.size())] += 1;
            }
            if (cluster_B[dest] != 0) {
                this->innerAndCutEdge[std::make_pair(cluster_B[src] , cluster_S[dest] + cluster_B.size())] += 1;
            }
        } 
    }

}

void StreamCluster::calculateDegree() {
    int count = 0;
    std::pair<int,int> edge(-1,-1);
    std::string inputGraphPath = config.inputGraphPath;
    TGEngine tgEngine(inputGraphPath,3997962,16539643);  
    // std::cout << "count :"  << count << std::endl;
    while (-1 != tgEngine.readline(edge)) {
        // int src = edge.first;
        // int dest = edge.second;
        count++;
        // degree[src] ++;
        // degree[dest] ++;
        // std::cout << "count :"  << count << std::endl;
    }
    
    std::cout << "count :"  << count << std::endl;
    std::cout << "End CalculateDegree" << std::endl;
    exit(-1);
}


int StreamCluster::getEdgeNum(int cluster1, int cluster2, std::string type) {
    if(innerAndCutEdge.find(std::make_pair(cluster1, cluster2)) != innerAndCutEdge.end()) {
        innerAndCutEdge[std::make_pair(cluster1, cluster2)];
    }
    return 0;
}

int StreamCluster::getEdgeNum(int cluster1, int cluster2) {
    if(innerAndCutEdge.find(std::make_pair(cluster1, cluster2)) != innerAndCutEdge.end()) {
        innerAndCutEdge[std::make_pair(cluster1, cluster2)];
    }
    return 0;
}
std::vector<int> StreamCluster::getClusterList() {
    return clusterList;
}

std::vector<int> StreamCluster::getCluster() {
    return cluster;
}

std::vector<int> StreamCluster::getDegree() {
    std::cout << "get degree..." << std::endl;
    return degree;
}

int StreamCluster::getClusterId(int id, std::string graphType) {
    if (graphType == "S")
        return cluster_S[id];
    return cluster_B[id];
}



std::vector<int> StreamCluster::getClusterList_B() {
    return clusterList_B;
}

std::vector<int> StreamCluster::getClusterList_S() {
    return clusterList_S;
}




int StreamCluster::getMaxVolume(){
    return maxVolume;
}










