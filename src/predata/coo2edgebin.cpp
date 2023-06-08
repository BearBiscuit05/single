#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <map>
#include <sstream>
#include <cassert>

void COO2edge(const std::string& csvFile,const std::string& binfile, std::vector<std::vector<int>>& src) {
    std::ifstream file(csvFile);
    std::ofstream outputFile(binfile, std::ios::binary);
    int startNode, endNode;
    char dot;
    while (file >> startNode >> dot >> endNode) {
        outputFile.write(reinterpret_cast<const char*>(&startNode), sizeof(startNode));
        outputFile.write(reinterpret_cast<const char*>(&endNode), sizeof(endNode));
    }
    file.close();
}