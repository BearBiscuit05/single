#include "readGraph.h"

ReadEngine::ReadEngine() {}

ReadEngine::ReadEngine(std::string graphPath)
{
    graphPath = graphPath;
    srcPath = graphPath + "srcList.bin";
    dstPath = graphPath + "dstList.bin";
    trainMaskPath = graphPath + "trainIDs.bin";

    srcFd = open(srcPath.c_str(), O_RDONLY);
    dstFd = open(dstPath.c_str(), O_RDONLY);
    tidsFd = open(trainMaskPath.c_str(), O_RDONLY);
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
}

int ReadEngine::readline(std::pair<int64_t, int64_t> &edge) {
    if (readPtr == edgeNUM){
        unmapBlock(srcAddr,chunkSize);
        unmapBlock(dstAddr,chunkSize);
        return -1;
    }
        
    if (readPtr % batch == 0){
        if (chunkSize != 0) {
            unmapBlock(srcAddr,chunkSize);
            unmapBlock(dstAddr,chunkSize);
        }
        loadingMmapBlock();
    }
    edge.first = srcAddr[readPtr%batch];
    edge.second = dstAddr[readPtr%batch];
    readPtr++;
    return 0;
}

int ReadEngine::readlines(std::vector<std::pair<int64_t, int64_t>> &edges,std::vector<int64_t>& eids,int& edgesNUM) {
    if (readPtr == edgeNUM)
        return -1;
    std::pair<int64_t, int64_t> edge;
    int i = 0;
    for ( ; i < edgesNUM ; i++) {
        if(-1 != this->readline(edge)) {
            eids[i] = this->readPtr - 1;
            edges[i] = edge;
        }
    }
    return i;
    
}

void ReadEngine::loadingMmapBlock() {
    chunkSize = std::min((long)readSize, srcLength - offset);
    srcAddr = static_cast<int64_t*>(mmap(nullptr, chunkSize, PROT_READ, MAP_SHARED, srcFd, offset));
    dstAddr = static_cast<int64_t*>(mmap(nullptr, chunkSize, PROT_READ, MAP_SHARED, dstFd, offset));
    if((srcAddr == MAP_FAILED) || (dstAddr == MAP_FAILED))
    {
        perror("mmap");
        close(srcFd);close(dstFd);close(tidsFd);
    }
    offset += chunkSize;
}

void ReadEngine::unmapBlock(int64_t* addr, off_t size) {
    munmap(addr, size);
}

void ReadEngine::readTrainIdx(std::vector<int64_t>& ids) {
    int tidsNUM = tidsLength / sizeof(int64_t);
    tidsAddr = static_cast<int64_t*>(mmap(nullptr, tidsLength, PROT_READ, MAP_SHARED, tidsFd, 0));
    ids.clear();
    ids.resize(tidsNUM);
    for (int i = 0; i < tidsNUM; ++i) {
        ids[i] = tidsAddr[i];
    }
}

