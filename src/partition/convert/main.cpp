#include "readGraph.h"


using namespace std;


int main() {
    // 3072441,10308445
    TGEngine tgEngine; 
    //char t = ''
    //tgEngine.convert2bin("/home/dzz/graphdataset/com-orkut/Dcom-orkut.ungraph.txt","edge.bin",'\t',false,"");
    tgEngine.convert_edgelist("/home/dzz/graphdataset/LJ/Dcom-lj.ungraph.txt","com.bin");
    
    return 0;
}
