#include "readGraph.h"


using namespace std;


int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <binfile_path> <num_nodes> <feat_len>" << std::endl;
        return 1;
    }

    // 获取命令行参数
    const char* binfilePath = argv[1];
    int numNodes = std::stoi(argv[2]);
    int featLen = std::stoi(argv[3]);

    // 在这里使用获取到的参数
    std::cout << "Binfile Path: " << binfilePath << std::endl;
    std::cout << "Number of Nodes: " << numNodes << std::endl;
    std::cout << "Feature Length: " << featLen << std::endl;
    // 3072441,10308445
    TGEngine tgEngine; 
    //char t = ''
    //tgEngine.convert2bin("/home/dzz/graphdataset/com-orkut/Dcom-orkut.ungraph.txt","edge.bin",'\t',false,"");
    
    // int64_t NUM_NODE=41652230;
    // int featLen = 300;
    // tgEngine.createBinfile("/raid/bear/dataset/twitter/feats_300.bin",NUM_NODE,featLen);

    // int64_t NUM_NODE=77741046;
    // int featLen = 300;
    tgEngine.createBinfile(binfilePath,numNodes,featLen);
    
    // int64_t NUM_NODE=105896555;
    // int featLen = 300;
    // tgEngine.createBinfile("/raid/bear/dataset/uk-2007-05/feats_300.bin",NUM_NODE,featLen);

    return 0;
}