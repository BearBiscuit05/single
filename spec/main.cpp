#include "readGraph.h"
#include "iostream"
#include <omp.h>
#define MAX_ITEMS 5000000    // 设置最大元素个数
#define P_ERROR 0.000001     // 设置误差

#define BATCH 1024
#define THREADNUM 8
int main() {
    // std::string graphPath = "/raid/bear/papers_bin/";
    // ReadEngine readengine(graphPath);
    // std::pair<int64_t,int64_t> edge(-1,-1);
    // int sum = 0;
    // while( -1 != readengine.readline(edge)){
    //     sum++;
    // }
    // std::cout << "one hop num :" << sum << std::endl;
    // std::cout << "all edges num :" << readengine.readPtr << std::endl;
    //std::string graphPath = "/home/bear/workspace/singleGNN/spec/edges.bin";
    // TGEngine tgEngine(graphPath,9498,153138);
    // int sum = 0;
    // std::pair<int,int> edge(-1,-1);
    // while( -1 != tgEngine.readline(edge)){
    //     std::cout << edge.first << " --> " << edge.second << std::endl;
    //     sum++;
    // }
    // std::cout << "one hop num :" << sum << std::endl;
    std::string inputfile = "/home/dzz/graphdataset/small.txt";
    std::string test="test.bin";
    TGEngine tgEngine;
    char delimiter = ' ';
    tgEngine.convert2bin(inputfile,test,delimiter,true);
    return 0;
}

// void FIXLINE(char *s)
// {
//     int len = (int)strlen(s) - 1;
//     if (s[len] == '\n')
//         s[len] = 0;
// }

// int main(){
//     std::string inputfile = "/home/dzz/graphdataset/graphpartition_smallgraph/Dmusae/Dmusae_DE.edges";
//     FILE *inf = fopen(inputfile.c_str(), "r");
//     size_t bytesread = 0;
//     size_t linenum = 0;
//     if (inf == NULL) {
//         std::cout << "Could not load:" << inputfile
//                    << ", error: " << strerror(errno) << std::endl;
//     }
//     std::ofstream fout("edges.bin");
//     std::cout << "Reading in edge list format!" << std::endl;
//     char s[1024];
//     while (fgets(s, 1024, inf) != NULL) {
//         linenum++;
//         if (linenum % 10000000 == 0) {
//             std::cout << "Read " << linenum << " lines, "
//                       << bytesread / 1024 / 1024. << " MB" << std::endl;
//         }
//         FIXLINE(s);
//         bytesread += strlen(s);
//         if (s[0] == '#')
//             continue; // Comment
//         if (s[0] == '%')
//             continue; // Comment

//         char delims[] = "\t, ";
//         char *t;
//         t = strtok(s, delims);
//         if (t == NULL) {
//             std::cout << "Input file is not in right format. "
//                        << "Expecting \"<from>\t<to>\". "
//                        << "Current line: \"" << s << "\"\n";
//         }
//         int from = atoi(t);
//         t = strtok(NULL, delims);
//         if (t == NULL) {
//             std::cout << "Input file is not in right format. "
//                        << "Expecting \"<from>\t<to>\". "
//                        << "Current line: \"" << s << "\"\n";
//         }
//         int to = atoi(t);

//         // if (from != to) {
//         //     std::cout << from << " --> " << to << std::endl;
//         // }
//         fout.write((char *)&from, sizeof(int));
//         fout.write((char *)&to, sizeof(int));
        
//     }
//     fclose(inf);
//     fout.close();
// }
