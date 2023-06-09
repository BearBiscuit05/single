#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <map>
#include <sstream>
#include <cassert>

void COO2edge(const std::string& txtFile,const std::string& binfile) {
    std::ofstream outputFile(binfile, std::ios::binary);
    std::ifstream file(txtFile);
    std::string line;
    std::getline(file, line);
    std::stringstream confss(line);
    std::vector<int> confData; // [nodeNUM,edgeNUM]
    std::string token;
    while (std::getline(confss, token, ',')) { 
        int value = std::stoi(token);
        confData.push_back(value);
    }
    std::map<int,std::vector<int>> graph;

    if (file) {   
        while (std::getline(file, line)) {
            int key;
            std::vector<int> values;
            std::stringstream ss(line);
            
            std::getline(ss, token, ',');
            key = std::stoi(token);

            while (std::getline(ss, token, ',')) {
                int value = std::stoi(token);
                values.push_back(value);
            }
            graph[-key] = values;
        }
        file.close();
        std::cout << "read txt file success." << std::endl;
    } else {
        std::cerr << "Failed to open input or output file." << std::endl;
    }
    
    file.close();
    for (auto it = graph.begin(); it != graph.end(); ++it) {
        int dst = it->first;
        auto srcs = it->second;
        for(int src : srcs) {
            outputFile.write(reinterpret_cast<const char*>(&src), sizeof(src));
            outputFile.write(reinterpret_cast<const char*>(&dst), sizeof(dst));
        }
    }
    outputFile.close();

}

int main(int argc, char* argv[]) {
    std::string txtFile = argv[1];
    std::string savePath = argv[2];
    COO2edge(txtFile,savePath);
    return 0;
}
