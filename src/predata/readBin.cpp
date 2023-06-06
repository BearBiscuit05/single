#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <cassert>

/*
    读取指定格式数据
    nodeNUM,edgeNUM,-srcid1,value1,value2,-srcid2,value1,value2...
*/
int main(int argc, char ** argv) {
    if(argc!=2) {
        std::cout << "Usage: " << argv[0] << " <file>" << std::endl;
        return 0;
    }
    FILE * fp = fopen64(argv[1],"r");
    assert(fp!=NULL);

    uint nodes, edges, rd = 0;
    rd += fread(&nodes, sizeof(uint), 1, fp);
    rd += fread(&edges, sizeof(uint), 1, fp);

    std::map<int,std::vector<int>> csr;

    std::cout << "* nodes=" << nodes << std::endl;
    std::cout << "* edges=" << edges << std::endl;
    // std::cout << "Building structure...";
    int nodeID = 0;
    uint ml = 0, al = 0;
    for(uint i=0;i<edges+nodes;i++) {
        int s;    
        rd += fread(&s, sizeof(int), 1, fp); 
        if(s<0) {
            nodeID = -s;
        }
        else {
            csr[nodeID].push_back(s);
        }
    }
    // std::cout << "reading..." << std::endl;
    // std::vector<int> it = csr[446871];
    // for(int i = 0 ; i < it.size() ; i++){
    //     std::cout << it[i] << " ";
    //     if (i % 5 == 0)
    //         std::cout << std::endl;
    // }
    return 0;
}
