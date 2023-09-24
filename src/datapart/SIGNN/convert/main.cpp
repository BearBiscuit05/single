#include "readGraph.h"


using namespace std;


int main(int argc, char* argv[]) {
    // if (argc != 4) {
    //     std::cerr << "Usage: " << argv[0] << " <binfile_path> <num_nodes> <feat_len>" << std::endl;
    //     return 1;
    // }

    TGEngine tgEngine; 
    //char t = ''
    //tgEngine.convert2bin("/home/dzz/graphdataset/com-orkut/Dcom-orkut.ungraph.txt","edge.bin",'\t',false,"");
    
    // int64_t NUM_NODE=41652230;
    // int featLen = 300;
    // tgEngine.createBinfile("/raid/bear/dataset/twitter/feats_300.bin",NUM_NODE,featLen);

    // int64_t NUM_NODE=77741046;
    // int featLen = 300;
    std::string infile = "/raid/bear/dataset/com_fr/com-friendster.ungraph.txt";
    std::string outer = "/raid/bear/dataset/com_fr/com_fr.bin";
    // int nodeNUM = 65608366;
    // int edgeNUM = 1806067135;
    tgEngine.convert2bin(infile,outer,' ',false,"");
    
    // int64_t NUM_NODE=105896555;
    // int featLen = 300;
    // tgEngine.createBinfile("/raid/bear/dataset/uk-2007-05/feats_300.bin",NUM_NODE,featLen);

    return 0;
}