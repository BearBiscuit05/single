#include "readGraph.h"

PartitionEngine::PartitionEngine() {}

PartitionEngine::PartitionEngine(std::string graphPath) : srcStream(graphPath + "srcList.bin",std::ios::binary),
    dstStream(graphPath + "dstList.bin",std::ios::binary),trainIdStream(graphPath + "trainIDs.bin",std::ios::binary)
{
    graphPath = graphPath;
    srcPath = graphPath + "srcList.bin";
    dstPath = graphPath + "dstList.bin";
    trainMaskPath = graphPath + "trainIDs.bin";
}

int PartitionEngine::readline(std::pair<int64_t, int64_t> &edge) {
    if (srcStream.read(reinterpret_cast<char*>(&edge.first), sizeof(int64_t)) && 
            dstStream.read(reinterpret_cast<char*>(&edge.second), sizeof(int64_t)))
        return 0;
    else
        return -1;
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