#include "StreamCluster.h"
#include <iostream>
#include <algorithm>
#include <fstream>

int _readStep(std::ifstream& fileStream,Edge& edge) {
    // std::cout << 666666666 << std::endl;
    // std::cout << (!fileStream) << std::endl;
    std::string line;
    if (std::getline(fileStream, line)) {
        // std::cout << line << std::endl;
        if (line.empty() || line[0] == '#')
            return _readStep(fileStream,edge);

        size_t tabPos = line.find('\t');
        int srcVId = std::stoi(line.substr(0, tabPos));
        int destVId = std::stoi(line.substr(tabPos + 1));

        if (srcVId == destVId)
            return _readStep(fileStream,edge);
        edge.srcVId = srcVId;
        edge.destVId = destVId;
        edge.weight = 1;
        return 0;
    }
    std::cout << "read end..." << std::endl;
    return -1; // Return an empty edge if end of file is reached
}


StreamCluster::StreamCluster() {}

StreamCluster::StreamCluster(Graph& graph, GlobalConfig& config) {
    this->config = config;
    this->cluster_B.resize(size_t(config.vCount),-1);
    this->cluster_S.resize(size_t(config.vCount),-1);

    this->volume_B.resize(size_t(0.1 * config.vCount),0);
    this->volume_S.resize(size_t(0.1 * config.vCount),0);
    this->graph = &graph;
    // Rest of the constructor implementation
    maxVolume = config.getMaxClusterVolume();
    degree.resize(config.vCount,0);
    // degree_B.resize(config.vCount,0);
    degree_S.resize(config.vCount,0);
    // std::cout << "end"<< std::endl;
    calculateDegree();
    // std::cout << "end"<< std::endl;
}

void StreamCluster::setCluster(std::vector<int> cluster) {
    this->cluster = cluster;
}

void StreamCluster::setDegree(std::vector<int> degree) {
    this->degree = degree;
}

// void StreamCluster::setVolume_S(std::unordered_map<int, int> volume_S) {
//     this->volume_S = volume_S;
// }

void StreamCluster::setClusterList(std::vector<int> clusterList) {
    this->clusterList = clusterList;
}

void StreamCluster::setClusterList_S(std::vector<int> clusterList_S) {
    this->clusterList_S = clusterList_S;
}

void StreamCluster::setClusterList_B(std::vector<int> clusterList_B) {
    this->clusterList_B = clusterList_B;
}

void StreamCluster::setMaxVolume(int maxVolume) {
    this->maxVolume = maxVolume;
}

void StreamCluster::setInnerAndCutEdge(std::unordered_map<int, std::unordered_map<int, int>> innerAndCutEdge) {
    innerAndCutEdge = innerAndCutEdge;
}


// void StreamCluster::setUpIndex() {
//     std::vector<std::pair<int, int>> sortList_B(volume_B.begin(), volume_B.end());
//     std::sort(sortList_B.begin(), sortList_B.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
//         return a.second > b.second;
//     });

//     for (const auto& entry : sortList_B) {
//         if (entry.second == 0) {
//             continue;
//         }
//         clusterList_B.push_back(entry.first);
//     }

//     volume_B.clear();

//     // Sort the volume of the cluster for clusterList_S
//     std::vector<std::pair<int, int>> sortList_S(volume_S.begin(), volume_S.end());
//     std::sort(sortList_S.begin(), sortList_S.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
//         return a.second > b.second;
//     });

//     for (const auto& entry : sortList_S) {
//         if (entry.second == 0) {
//             continue;
//         }
//         clusterList_S.push_back(entry.first);
//     }

//     volume_S.clear();
// }

void StreamCluster::startStreamCluster() {
    double averageDegree = config.getAverageDegree();
    int clusterID_B = 0;
    int clusterID_S = 0;
    std::cout << "start read Streaming Clustring..." << std::endl;
    std::string inputGraphPath = config.inputGraphPath;
    std::ifstream tmp(inputGraphPath);
    std::string line;
    Edge edge(-1,-1,-1);  
    while (-1 != _readStep(tmp,edge)) {
        int src = edge.getSrcVId();
        int dest = edge.getDestVId();
        if (degree[src] >= config.getTao() * averageDegree && degree[dest] >= config.getTao() * averageDegree) {
            if (cluster_B[src] == -1) {
                cluster_B[src] = clusterID_B++;
            }
            if (cluster_B[dest] == -1) {
                cluster_B[dest] = clusterID_B++;
            }

            if (cluster_B[src] >= volume_B.size() || cluster_B[dest] >= volume_B.size()) {
                volume_B.resize(volume_B.size() + 0.1 * config.vCount, 0);
            }
            // auto src_it = volume_B[cluster_B[src]];
            // if (src_it == volume_B.end()) {
            //     volume_B[cluster_B[src]] = 0;
            // }
            // auto dest_it = volume_B.find(cluster_B[dest]);
            // if ( dest_it == volume_B.end()) {
            //     volume_B[cluster_B[dest]] = 0;
            // }
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
        // if (volume_S[i] == 0) {
        //     continue;
        // }
        clusterList_S.push_back(i + cluster_B.size());
    }
    volume_S.clear();  
    // std::vector<std::pair<int, int>> sortList_S(volume_S.begin(), volume_S.end());

    // // Add non-zero keys to clusterList_S
    // for (const auto& entry : sortList_S) {
    //     if (entry.second != 0) {
    //         clusterList_S.push_back(entry.first);
    //     }
    // }
    // volume_S.clear();

    
    // this->volume_S.resize(size_t(0.1 * config.vCount),0);
    this->config.clusterBSize = cluster_B.size();

    std::cout << "!"<<cluster_B.size() << std::endl;
}

// void StreamCluster::startSteamClusterB() {
//     double averageDegree = config.getAverageDegree();
//     int clusterID_B = 1;
//     std::cout << "start read CB..." << std::endl;
//     std::string inputGraphPath = config.inputGraphPath;
//     std::ifstream tmp(inputGraphPath);
//     std::string line;
//     Edge edge(-1,-1,-1);

//     while (-1 != _readStep(tmp,edge)) {
//         int src = edge.getSrcVId();
//         int dest = edge.getDestVId();
//         if (degree[src] >= config.getTao() * averageDegree && degree[dest] >= config.getTao() * averageDegree) {
//             if (cluster_B[src] == 0) {
//                 cluster_B[src] = clusterID_B++;
//             }
//             if (cluster_B[dest] == 0) {
//                 cluster_B[dest] = clusterID_B++;
//             }
//             // degree_B[src]++;
//             // degree_B[dest]++;
//             // update volume

//             //TODO:resize
//             auto src_it = volume_B.find(cluster_B[src]);
//             if ( src_it == volume_B.end()) {
//                 volume_B[cluster_B[src]] = 0;
//             }
//             auto dest_it = volume_B.find(cluster_B[dest]);
//             if ( dest_it == volume_B.end()) {
//                 volume_B[cluster_B[dest]] = 0;
//             }
//             volume_B[cluster_B[src]]++;
//             volume_B[cluster_B[dest]]++;

//             if (volume_B[cluster_B[src]] >= maxVolume) {
//                 volume_B[cluster_B[src]] -= degree[src];
//                 cluster_B[src] = clusterID_B++;
//                 volume_B[cluster_B[src]] = degree[src];
//             }

//             if (volume_B[cluster_B[dest]] >= maxVolume) {
//                 volume_B[cluster_B[dest]] -= degree[dest];
//                 cluster_B[dest] = clusterID_B++;
//                 volume_B[cluster_B[dest]] = degree[dest];
//             }
//             if (volume_B[cluster_B[src]] >= maxVolume || volume_B[cluster_B[dest]] >= maxVolume)
//                 continue;

//             // int minVid = (volume_B[cluster_B[src]] < volume_B[cluster_B[dest]] ? src : dest);
//             // int maxVid = (src == minVid ? dest : src);
//             // if ((volume_B[cluster_B[maxVid]] + degree_B[minVid]) <= maxVolume) {
//             //     volume_B[cluster_B[maxVid]] += degree_B[minVid];
//             //     volume_B[cluster_B[minVid]] -= degree_B[minVid];
//             //     if (volume_B[cluster_B[minVid]] == 0)
//             //         volume_B.erase(cluster_B[minVid]);
//             //     cluster_B[minVid] = cluster_B[maxVid];
//             // }
//         }
//     }
    
//     //TODO
//     std::vector<std::pair<int, int>> sortList_B(volume_B.begin(), volume_B.end());

//     for (const auto& entry : sortList_B) {
//         if (entry.second == 0) {
//             continue;
//         }
//         clusterList_B.push_back(entry.first);
//     }
//     volume_B.clear();
    
// }

// void StreamCluster::startSteamClusterS() {
//     double averageDegree = config.getAverageDegree();
//     int clusterID_S = 1;
//     std::string inputGraphPath = config.inputGraphPath;
//     std::ifstream tmp(inputGraphPath);
//     Edge edge(-1,-1,-1);
//     std::cout << "start read CS..." << std::endl;
//     while (-1 != _readStep(tmp,edge)) {
//         int src = edge.getSrcVId();
//         int dest = edge.getDestVId();
//         // allocate cluster
//         if (!(degree[src] >= config.getTao() * averageDegree && degree[dest] >= config.getTao() * averageDegree)) {
//             if (cluster_S[src] == 0) {
//                 cluster_S[src] = clusterID_S++;
//             }
//             if (cluster_S[dest] == 0) {
//                 cluster_S[dest] = clusterID_S++;
//             }
//             degree_S[src]++;
//             degree_S[dest]++;

//             // update volume
//             auto src_it = volume_S.find(cluster_S[src]);
//             if ( src_it == volume_S.end()) {
//                 volume_S[cluster_S[src]] = 0;
//             }
//             auto dest_it = volume_S.find(cluster_S[dest]);
//             if ( dest_it == volume_S.end()) {
//                 volume_S[cluster_S[dest]] = 0;
//             }
//             volume_S[cluster_S[src]]++;
//             volume_S[cluster_S[dest]]++;

//             // if (volume_S[cluster_S[src]] >= maxVolume) {
//             //     volume_S[cluster_S[src]] -= degree_S[src];
//             //     cluster_S[src] = clusterID_S++;
//             //     volume_S[cluster_S[src]] = degree_S[src];
//             // }

//             // if (volume_S[cluster_S[dest]] >= maxVolume) {
//             //     volume_S[cluster_S[dest]] -= degree_S[dest];
//             //     cluster_S[dest] = clusterID_S++;
//             //     volume_S[cluster_S[dest]] = degree_S[dest];
//             // }

//             if (volume_S[cluster_S[src]] >= maxVolume || volume_S[cluster_S[dest]] >= maxVolume)
//                 continue;

//             int minVid = (volume_S[cluster_S[src]] < volume_S[cluster_S[dest]] ? src : dest);
//             int maxVid = (src == minVid ? dest : src);

//             if ((volume_S[cluster_S[maxVid]] + degree_S[minVid]) <= maxVolume) {
//                 volume_S[cluster_S[maxVid]] += degree_S[minVid];
//                 volume_S[cluster_S[minVid]] -= degree_S[minVid];
//                 if (volume_S[cluster_S[minVid]] == 0)
//                     volume_S.erase(cluster_S[minVid]);
//                 cluster_S[minVid] = cluster_S[maxVid];
//             }
//         }
//     }

//     // Sort the volume_S map by value (second element) in descending order
//     // TODO : 一定要排序?
//     std::vector<std::pair<int, int>> sortList_S(volume_S.begin(), volume_S.end());
//     std::sort(sortList_S.begin(), sortList_S.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
//         return a.second > b.second;
//     });

//     // Add non-zero keys to clusterList_S
//     for (const auto& entry : sortList_S) {
//         if (entry.second != 0) {
//             clusterList_S.push_back(entry.first);
//         }
//     }
//     volume_B.clear();
// }

void StreamCluster::computeHybridInfo() {

    Edge edge(-1,-1,-1);
    std::string inputGraphPath = config.inputGraphPath;
    std::ifstream tmp(inputGraphPath);
 
    while (-1 != _readStep(tmp,edge)) {
        int src = edge.getSrcVId();
        int dest = edge.getDestVId();
        int oldValue = 0;
        if (degree[src] >= config.tao * config.getAverageDegree() && degree[dest] >= config.tao * config.getAverageDegree()) {
            this->innerAndCutEdge[std::make_pair(cluster_B[src], cluster_B[dest])] += 1;
        } else {
            // if(cluster_S[src] + cluster_B.size() == 16395){
            //     std::cout << "YES" << std::endl;
            // }
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
    //First Time
    Edge edge(-1,-1,-1);
    int count = 0;
   
    while(-1 != graph->readStep(edge)) {
        int src = edge.getSrcVId();
        int dest = edge.getDestVId();
        count++;
        degree[src] ++;
        degree[dest] ++;
    }
    std::cout << "count :"  << count << std::endl;
    std::cout << "End CalculateDegree" << std::endl;
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










