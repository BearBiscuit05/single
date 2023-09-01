#include "readGraph.h"

PartitionEngine::PartitionEngine() {}

PartitionEngine::PartitionEngine(std::string graphPath)
{
    graphPath = graphPath;
    srcPath = graphPath + "srcList.bin";
    dstPath = graphPath + "dstList.bin";
    trainMaskPath = graphPath + "trainIDs.bin";

    int srcFd = open(srcPath.c_str(), O_RDONLY);
    int dstFd = open(dstPath.c_str(), O_RDONLY);
    int tidsFd = open(trainMaskPath.c_str(), O_RDONLY);
    if ((srcFd == -1) || (dstFd == -1) || (tidsFd == -1) ) {
        perror("open");
    }

    struct stat sb;
    if (fstat(srcFd, &sb) == -1) {
        perror("fstat");close(srcFd);
    }
    srcLength = sb.st_size;
    if (fstat(dstFd, &sb) == -1) {
        perror("fstat");close(dstFd);
    }
    dstLength = sb.st_size;
    if (fstat(tidsFd, &sb) == -1) {
        perror("fstat");close(tidsFd);
    }
    tidsLength = sb.st_size;
    edgeNUM = srcLength / sizeof(int64_t);

    srcAddr = static_cast<int64_t*>(mmap(nullptr, srcLength, PROT_READ, MAP_SHARED, srcFd, 0));
    dstAddr = static_cast<int64_t*>(mmap(nullptr, dstLength, PROT_READ, MAP_SHARED, dstFd, 0));
    tidsAddr = static_cast<int64_t*>(mmap(nullptr, tidsLength, PROT_READ, MAP_SHARED, tidsFd, 0));
     
    if((srcAddr == MAP_FAILED) || (dstAddr == MAP_FAILED) || (tidsAddr == MAP_FAILED))
    {
        perror("mmap");
        close(srcFd);close(dstFd);close(tidsFd);
    }
}

int PartitionEngine::readline(std::pair<int64_t, int64_t> &edge) {
    if (readPtr == edgeNUM) 
        return -1;
    if (readPtr % 1024 == 0){
        if(readPtr+1024 < edgeNUM){
            srcCache.assign(srcAddr+readPtr,srcAddr+readPtr+1024);
        }
        else {
            srcCache.assign(srcAddr+readPtr,srcAddr+edgeNUM);
        }

    }
    edge.first = srcAddr[readPtr%1024];
    edge.second = dstAddr[readPtr%1024];
    readPtr++;
    return 0;
}


int main() {
    std::string graphPath = "/raid/bear/papers_bin/";
    PartitionEngine engine(graphPath);
    std::pair<int64_t, int64_t> edge;
    int64_t count = 0;
    while(-1 != engine.readline(edge)){
        count++;
    }
    // for (int i = 0 ; i < 10 ; i++) {
        
        
    //     // std::cout << edge.first << " --> " << edge.second << std::endl;
    // }
    std::cout << "edgeNUM :" << count << std::endl;
    return 0;
}