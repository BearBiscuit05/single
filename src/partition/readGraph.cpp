#include "readGraph.h"

PartitionEngine::PartitionEngine() {}

PartitionEngine::PartitionEngine(std::string graphPath)
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

int PartitionEngine::readline(std::pair<int64_t, int64_t> &edge) {
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

void PartitionEngine::loadingMmapBlock() {
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

void PartitionEngine::unmapBlock(int64_t* addr, off_t size) {
    munmap(addr, size);
}

int main() {
    std::string graphPath = "/raid/bear/papers_bin/";
    PartitionEngine engine(graphPath);
    std::pair<int64_t, int64_t> edge;
    int64_t count = 0;
    while(-1 != engine.readline(edge)){
        count++;
    }
    std::cout << "edgeNUM :" << count << std::endl;
    return 0;
}