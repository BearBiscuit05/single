#include "StreamCluster.h"
#include <iostream>
#include <algorithm>
#include <fstream>
#include "threadpool.h"




StreamCluster::StreamCluster() :c(0.1, 0.01) {}

StreamCluster::StreamCluster(GlobalConfig& config) 
    : config(config),c(0.1, 0.01){
    this->cluster_B.resize(size_t(config.vCount),-1);
    this->cluster_S.resize(size_t(config.vCount),-1);
    this->volume_B.resize(size_t(config.vCount),0);
    this->volume_S.resize(size_t(config.vCount),0);
    maxVolume = config.getMaxClusterVolume();
    degree.resize(config.vCount,0);
    degree_S.resize(config.vCount,0);
    calculateDegree();
    Triplet tmp = {-1,-1,0};
    this->cacheData.resize(BATCH,tmp);
}

void addSketchKV(CountMinSketch* c,std::string str) {
    c->update(str.c_str(), 1);
}

void StreamCluster::Start() {
    producerThread = std::thread(&StreamCluster::Producer, this);
    consumerThread = std::thread(&StreamCluster::Consumer, this);
}

void StreamCluster::Stop() {
    producerThread.join();
    consumerThread.join();
}


void StreamCluster::Producer() {
    int clusterID_B = 0;
    int clusterID_S = 0;
    std::pair<int,int> edge(-1,-1);
    double averageDegree = config.getAverageDegree();
    int clusterNUM = config.vCount;
    std::string inputGraphPath = config.inputGraphPath;
    TGEngine tgEngine(inputGraphPath,NODENUM,EDGENUM);
    std::string key="";
    std::vector<std::string> s_key(BATCH);
    
    int lineNUM = 0;
    while (-1 != tgEngine.readline(edge)) {
        // if (lineNUM + 4 >= BATCH) {
        //     std::unique_lock<std::mutex> lock(mtx);
        //     for (int li = 0 ; li < lineNUM ; li++)
        //         buffer.push(s_key[li]);
        //     cv.notify_all();
        //     lineNUM = 0;
        // }

        int src = edge.first;
        int dest = edge.second;
        if (degree[src] >= config.tao * averageDegree && degree[dest] >= config.tao * averageDegree) {
            this->isInB[tgEngine.readPtr/2] = true;
            auto& com_src = cluster_B[src];
            auto& com_dest = cluster_B[dest];
            if(com_src == -1) {
                com_src = clusterID_B;
                volume_B[com_src] += degree[src];
                ++clusterID_B;
            }
            if(com_dest == -1) {
                com_dest = clusterID_B;
                volume_B[com_dest] += degree[src];
                ++clusterID_B;
            }
            auto& vol_src = volume_B[com_src];
            auto& vol_dest = volume_B[com_dest];

            auto real_vol_src = vol_src - degree[src];
            auto real_vol_dest = vol_dest - degree[dest];

            if (real_vol_src <= real_vol_dest && vol_dest + degree[src] <= maxVolume) {
                vol_src -= degree[src];
               	vol_dest += degree[src];
               	cluster_B[src] = cluster_S[dest];
            } else if (real_vol_dest <= real_vol_src && vol_src + degree[dest] <= maxVolume) {
                vol_src -= degree[dest];
               	vol_dest += degree[dest];
               	cluster_B[dest] = cluster_B[src];
            }
            key = std::to_string(cluster_B[src]) + "," + std::to_string(cluster_B[dest]);
            this->c.update(key.c_str(), 1);
            //s_key[lineNUM++] = key;
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
            if (volume_S[cluster_S[src]] < maxVolume && volume_S[cluster_S[dest]] < maxVolume) {
                int minVid = (volume_S[cluster_S[src]] < volume_S[cluster_S[dest]] ? src : dest);
                int maxVid = (src == minVid ? dest : src);
                if ((volume_S[cluster_S[maxVid]] + degree_S[minVid]) <= maxVolume) {
                    volume_S[cluster_S[maxVid]] += degree_S[minVid];
                    volume_S[cluster_S[minVid]] -= degree_S[minVid];
                    cluster_S[minVid] = cluster_S[maxVid];
                }    
                key = std::to_string(cluster_S[src] + config.vCount) + "," + std::to_string(cluster_S[dest] + config.vCount);
                //s_key[lineNUM++] = key;
                this->c.update(key.c_str(), 1);
                if (cluster_B[src] !=-1) {
                    key = std::to_string(cluster_B[dest]) + "," + std::to_string(cluster_S[src] + config.vCount);
                    //s_key[lineNUM++] = key;
                    this->c.update(key.c_str(), 1);
                }
                if (cluster_B[dest] != -1) {
                    key = std::to_string(cluster_B[src]) + "," + std::to_string(cluster_S[dest] + config.vCount);
                    //s_key[lineNUM++] = key;
                    this->c.update(key.c_str(), 1);
                }   
            } else {
                continue;
            }
        }
    }

    // std::unique_lock<std::mutex> lock(mtx);
    // for (int li = 0 ; li < lineNUM ; li++)
    //     buffer.push(s_key[li]);
    // buffer.push("-1");
    // cv.notify_all();
    // lineNUM = 0;
}

void StreamCluster::Consumer() {  
    std::string key;
    int ssize;
    int loop;
    while (true) {
        std::unique_lock<std::mutex> lock(mtx);
        while (buffer.empty()) {
            cv.wait(lock);
        }
        ssize = buffer.size();
        loop = std::min(ssize,BATCH);
        for (int i = 0 ; i < loop ; i++) {
            key = buffer.front();
            buffer.pop();
            if (key == "-1")
                return;
            this->c.update(key.c_str(), 1);
        }
        cv.notify_all();
    }

    
}

void StreamCluster::startStreamCluster() {   
    std::cout << "start read Streaming Clustring..." << std::endl;
    this->isInB.resize(config.eCount,false);
    
    Producer();

    std::vector<std::pair<uint64_t, uint32_t>> sorted_communities;
    for (size_t i = 0; i < volume_B.size(); ++i)
    {
        if (volume_B[i] != 0)
            sorted_communities.emplace_back(volume_B[i], i);
    }
    volume_B.clear();  
    std::sort(sorted_communities.rbegin(), sorted_communities.rend()); // sort in descending order

    for (int i = 0; i < sorted_communities.size(); ++i) {
        clusterList_B.push_back(sorted_communities[i].second);
    }
    
    sorted_communities.clear();
    // for (int i = 0; i < volume_B.size(); ++i) {
    //     if (volume_B[i] != 0)
    //         clusterList_B.push_back(i);
    // }
    // volume_B.clear();  

    for (size_t i = 0; i < volume_S.size(); ++i)
    {
        if (volume_S[i] != 0)
            sorted_communities.emplace_back(volume_S[i], i);
    }
    volume_S.clear(); 
    std::sort(sorted_communities.rbegin(), sorted_communities.rend()); // sort in descending order

    for (int i = 0; i < sorted_communities.size(); ++i) {
        clusterList_S.push_back(sorted_communities[i].second + config.vCount);
    }
     
    sorted_communities.clear();        
    
    // for (int i = 0; i < volume_S.size(); ++i) {
    //     if (volume_S[i] != 0)
    //         clusterList_S.push_back(i + config.vCount);
    // }
    // volume_S.clear();  
    // this->config.clusterBSize = config.vCount;
    // computeHybridInfo();
}


void StreamCluster::startStreamCluster_MAP() {
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
    int cachePtr = 0;
    std::vector<std::unordered_map<std::string , int>> maplist;
    for (int i = 0 ; i < THREADNUM ; i++) {
        std::unordered_map<std::string , int> mapTmp;
        maplist.emplace_back(mapTmp);
    }
    while (-1 != tgEngine.readline(edge)) {
        if (cachePtr + 3 >= BATCH) {
            mergeMap(maplist,cachePtr);
        }
        int src = edge.first;
        int dest = edge.second;
        if (degree[src] >= config.tao * averageDegree && degree[dest] >= config.tao * averageDegree) {
            this->isInB[tgEngine.readPtr/2] = true;
            auto& com_src = cluster_B[src];
            auto& com_dest = cluster_B[dest];
            if(com_src == -1) {
                com_src = clusterID_B;
                volume_B[com_src] += degree[src];
                ++clusterID_B;
            }
            if(com_dest == -1) {
                com_dest = clusterID_B;
                volume_B[com_dest] += degree[src];
                ++clusterID_B;
            }
            auto& vol_src = volume_B[com_src];
            auto& vol_dest = volume_B[com_dest];

            auto real_vol_src = vol_src - degree[src];
            auto real_vol_dest = vol_dest - degree[dest];

            if (real_vol_src <= real_vol_dest && vol_dest + degree[src] <= maxVolume) {
                vol_src -= degree[src];
               	vol_dest += degree[src];
               	cluster_B[src] = cluster_S[dest];
            } else if (real_vol_dest <= real_vol_src && vol_src + degree[dest] <= maxVolume) {
                vol_src -= degree[dest];
               	vol_dest += degree[dest];
               	cluster_B[dest] = cluster_B[src];
            }
            this->cacheData[cachePtr].src = src;
            this->cacheData[cachePtr++].dst = dest;
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
            if (volume_S[cluster_S[src]] < maxVolume && volume_S[cluster_S[dest]] < maxVolume) {
                int minVid = (volume_S[cluster_S[src]] < volume_S[cluster_S[dest]] ? src : dest);
                int maxVid = (src == minVid ? dest : src);
                if ((volume_S[cluster_S[maxVid]] + degree_S[minVid]) <= maxVolume) {
                    volume_S[cluster_S[maxVid]] += degree_S[minVid];
                    volume_S[cluster_S[minVid]] -= degree_S[minVid];
                    cluster_S[minVid] = cluster_S[maxVid];
                }    
                this->cacheData[cachePtr].src = src;
                this->cacheData[cachePtr].dst = dest;
                this->cacheData[cachePtr++].flag = 1;
                if (cluster_B[src] !=-1) {
                    this->cacheData[cachePtr].src = src;
                    this->cacheData[cachePtr].dst = dest;
                    this->cacheData[cachePtr++].flag = 2;
                }
                if (cluster_B[dest] != -1) {
                    this->cacheData[cachePtr].src = src;
                    this->cacheData[cachePtr].dst = dest;
                    this->cacheData[cachePtr++].flag = 3;
                } 
            } else {
                continue;
            }
        }
    }
    mergeMap(maplist,cachePtr);
    for (int i = 1 ;  i < THREADNUM ; i++) {
        for(auto& m : maplist[i]) {
            maplist[0][m.first] += m.second;
        }
    }

    this->innerAndCutEdge = std::move(maplist[0]);
    maplist = std::vector<std::unordered_map<std::string , int>>();

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

void StreamCluster::mergeMap(std::vector<std::unordered_map<std::string , int>>& maplist,int& cachePtr) {
#pragma omp parallel for
    for (int i = 0 ;  i < cachePtr ; i++) {
        int flag = this->cacheData[i].flag;
        int tid = omp_get_thread_num();
        if(flag == 0) {
            maplist[tid][std::to_string(this->cluster_B[this->cacheData[i].src]) + "," + std::to_string(this->cluster_B[this->cacheData[i].dst])] += 1;
        } else if (flag == 1) {
            maplist[tid][std::to_string(this->cluster_S[this->cacheData[i].src] + this->config.vCount) + "," + std::to_string(this->cluster_S[this->cacheData[i].dst] + this->config.vCount)] += 1;
        } else if (flag == 2) {
            maplist[tid][std::to_string(this->cluster_S[this->cacheData[i].src] + this->config.vCount) + "," + std::to_string(this->cluster_B[this->cacheData[i].dst])] += 1;
        } else {
            maplist[tid][std::to_string(this->cluster_B[this->cacheData[i].src]) + "," + std::to_string(this->cluster_S[cacheData[i].dst] +  this->config.vCount)] += 1;
        }
    }
    cachePtr = 0;
}

void StreamCluster::computeHybridInfo() {
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
            std::string t = std::to_string(cluster_B[src]) + "," + std::to_string(cluster_B[dest]);
            // std::cout << t << std::endl;
            this->c.update(t.c_str(), 1);
        } else {
            std::string t = std::to_string(cluster_S[src] + config.vCount) + "," + std::to_string(cluster_S[dest] + config.vCount);
            if (cluster_B[src] !=-1) {
                t = std::to_string(cluster_B[dest]) + "," + std::to_string(cluster_S[src] + config.vCount);
            }
            if (cluster_B[dest] != -1) {
                t = std::to_string(cluster_B[src]) + "," + std::to_string(cluster_S[dest] + config.vCount);
            }   
        } 
    }
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
    // std::string index = std::to_string(cluster1) + "," + std::to_string(cluster2);
    // return this->c.estimate(index.c_str());

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

void StreamCluster::outputClusterSizeInfo() {
    std::string filename = "headcluster.csv";
    // std::string filenameTail = "tailcluster.csv";
    std::unordered_map<uint32_t, uint32_t> mp;
    for (int i = 0; i < this->volume_B.size(); ++i) {
        if (this->volume_B[i] != 0)
            mp[this->volume_B[i]]++;
    }
    std::ofstream outputFile(filename);
    int j = 0;
    for (auto &t : mp) {
        outputFile << t.first << "," << t.second;
        if (j < mp.size() - 1) {
            outputFile << "\n";
        }
        j++;
    }
    // 关闭文件
    outputFile.close();

    filename = "tailcluster.csv";
    mp.clear();
    for (int i = 0; i < this->volume_S.size(); ++i) {
        if (this->volume_S[i] != 0)
            mp[this->volume_S[i]]++;
    }
    std::ofstream outputFile2(filename);
    j = 0;
    for (auto &t : mp) {
        outputFile2 << t.first << "," << t.second;
        if (j < mp.size() - 1) {
            outputFile2 << "\n";
        }
        j++;
    }
    // 关闭文件
    outputFile2.close();
    std::cout << "end!!!!!!" << std::endl;
}












