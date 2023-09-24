#include "Partitioner.h"


Partitioner::Partitioner() {}

Partitioner::Partitioner(StreamCluster& streamCluster, GlobalConfig config)
    : streamCluster(&streamCluster), config(config) {
    this->gameRoundCnt_hybrid = 0;
    this->gameRoundCnt_inner = 0;
    partitionLoad.resize(config.partitionNum);
    std::cout << config.vCount << std::endl;
    v2p.resize(config.vCount, std::vector<char>(config.partitionNum));
    clusterPartition.resize(2*config.vCount, 0);
   }

void Partitioner::findPid_B(std::vector<std::pair<int,int>>& ids,std::vector<int>& pids,int& ptr) {
    double maxLoad = static_cast<double>(config.eCount) / config.partitionNum * config.alpha;
#pragma omp parallel for
    for(int index = 0 ; index < ptr ; index++) {
        int src = ids[index].first;
        int dst = ids[index].second;
        int srcPartition = clusterPartition[streamCluster->cluster_B[src]];
        int destPartition = clusterPartition[streamCluster->cluster_B[dst]];
        //int edgePartition = -1;
        if (partitionLoad[srcPartition] > maxLoad && partitionLoad[destPartition] > maxLoad) {
            for (int i = 0; i < config.partitionNum; i++) {
                if (partitionLoad[i] <= maxLoad) {
                    // edgePartition = i;
                    // srcPartition = i;
                    // destPartition = i;
                    pids[index] = i;
                    break;
                }
            }
        } else if (partitionLoad[srcPartition] > partitionLoad[destPartition]) {
            pids[index] = destPartition;
            // edgePartition = destPartition;
            // srcPartition = destPartition;
        } else {
            pids[index] = srcPartition;
            // edgePartition = srcPartition;
            // destPartition = srcPartition;
        }
    }
}

void Partitioner::findPid_S(std::vector<std::pair<int,int>>& ids,std::vector<int>& pids,int& ptr) {
    double maxLoad = static_cast<double>(config.eCount) / config.partitionNum * config.alpha;
#pragma omp parallel for    
    for(int index = 0 ; index < ptr ; index++) {
        int src = ids[index].first;
        int dst = ids[index].second;
        int srcPartition = clusterPartition[streamCluster->cluster_S[src]];
        int destPartition = clusterPartition[streamCluster->cluster_S[dst]];
        //int edgePartition = -1;
        if (partitionLoad[srcPartition] > maxLoad && partitionLoad[destPartition] > maxLoad) {
            for (int i = config.partitionNum - 1; i >= 0; i--) {
                if (partitionLoad[i] <= maxLoad) {
                    //edgePartition = i;
                    pids[index] = i;
                    // srcPartition = i;
                    // destPartition = i;
                    break;
                }
            }
        } else if (partitionLoad[srcPartition] > partitionLoad[destPartition]) {
            //edgePartition = destPartition;
            pids[index] = destPartition;
            // srcPartition = destPartition;
        } else {
            //edgePartition = srcPartition;
            pids[index] = srcPartition;
            // destPartition = srcPartition;
        }
    }
    
}

void Partitioner::performStep() {
    double maxLoad = static_cast<double>(config.eCount) / config.partitionNum * config.alpha;
    std::string inputGraphPath = config.inputGraphPath;
    std::pair<int,int> edge(-1,-1);
    TGEngine tgEngine(inputGraphPath,NODENUM,EDGENUM);  
    //std::vector<std::pair<int,int>> partCache(EDGENUM,edge);
    int partPtr = 0;
    std::vector<std::pair<int,int>> b_ids(P_BATCH,edge);
    std::vector<std::pair<int,int>> s_ids(P_BATCH,edge);
    std::vector<int> b_pid(P_BATCH,0);
    std::vector<int> s_pid(P_BATCH,0);
    int b_ptr = 0;
    int s_ptr = 0;
    while (-1 != tgEngine.readline(edge)) {
        int src = edge.first;
        int dest = edge.second;
        if (this->streamCluster->isInB[tgEngine.readPtr/2]) {
            if (b_ptr >= P_BATCH) {
                findPid_B(b_ids,b_pid,b_ptr);
                setPid(b_ids,b_pid,b_ptr);
                b_ptr = 0;
            }
            b_ids[b_ptr].first = src;
            b_ids[b_ptr++].second = dest;
        } else {
            if (s_ptr >= P_BATCH) {
                findPid_S(s_ids,s_pid,s_ptr);
                setPid(s_ids,s_pid,s_ptr);
                s_ptr=0;
            }
            s_ids[s_ptr].first = src;
            s_ids[s_ptr++].second = dest;
        }
    }
    findPid_B(b_ids,b_pid,b_ptr);
    setPid(b_ids,b_pid,b_ptr);
    findPid_S(s_ids,s_pid,s_ptr);
    setPid(s_ids,s_pid,s_ptr);
// #pragma omp parallel for shared(v2p)
//     for (int i = 0 ;  i < EDGENUM ; i++) {
//         v2p[this->streamCluster->cacheData[i].src][partCache[i].first] = true;
//         v2p[this->streamCluster->cacheData[i].dst][partCache[i].second] = true;
//     }
//     // std::cout << "end" << std::endl;
}

void Partitioner::setPid(std::vector<std::pair<int,int>>& ids,std::vector<int>& pids,int& ptr) {
    for (int i = 0 ; i < ptr ; i++) {
        partitionLoad[pids[i]]++;
        v2p[ids[i].first][pids[i]] = 1;
        v2p[ids[i].second][pids[i]] = 1;
    }

}



double Partitioner::getReplicateFactor() {
    int replicateTotal = 0;
    for (int i = 0; i < config.vCount; i++) {
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
    int batchSize = config.batchSize;
    std::vector<int> clusterList_B = streamCluster->getClusterList_B();
    std::vector<int> clusterList_S = streamCluster->getClusterList_S();
    int taskNum_B = (clusterList_B.size() + batchSize - 1) / batchSize;
    int taskNum_S = (clusterList_S.size() + batchSize - 1) / batchSize;
    int minTaskNUM = std::min(taskNum_B,taskNum_S);
    int leftTaskNUM = std::abs(taskNum_B - taskNum_S);

    
    std::cout << taskNum_B << " " << taskNum_S << std::endl;
    // ClusterGameTask cgt = ClusterGameTask(streamCluster,this->clusterPartition);
    std::vector<ClusterGameTask*> cgt_list(THREADNUM);
    for (int i = 0 ; i < THREADNUM ; i++) {
        ClusterGameTask* cgt = new ClusterGameTask((*streamCluster),this->clusterPartition);
        cgt_list[i] = cgt;
    }

#pragma omp parallel for
    for (int i = 0; i < minTaskNUM; i++) {
        int ompid = omp_get_thread_num();
        cgt_list[ompid]->resize_hyper("hybrid",i);
        cgt_list[ompid]->call();
        this->gameRoundCnt_hybrid += cgt_list[ompid]->roundCnt;
        // cgt.resize("hybrid",i);
        // cgt.call();
    }

    std::cout << "hyper end" <<std::endl;
    if (taskNum_B > taskNum_S) {
#pragma omp parallel for  
        for (int i = 0 ; i < leftTaskNUM; ++i) {
            int ompid = omp_get_thread_num();
            cgt_list[ompid]->resize("B",i+minTaskNUM);
            cgt_list[ompid]->call();
            this->gameRoundCnt_inner += cgt_list[ompid]->roundCnt;
            // cgt.resize("B",i+minTaskNUM);
            // cgt.call();
        }
    } else {
#pragma omp parallel for  
        for (int i = 0 ; i < leftTaskNUM; ++i) {
            int ompid = omp_get_thread_num();
            cgt_list[ompid]->resize("S",i+minTaskNUM);
            cgt_list[ompid]->call();
            this->gameRoundCnt_inner += cgt_list[ompid]->roundCnt;
            // cgt.resize("S",i+minTaskNUM);
            // cgt.call();
        }
    }


}

