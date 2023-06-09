#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <map>
#include <sstream>
#include <cassert>

void writeBinFile(std::string output_file,std::vector<int>& vec) {
    std::ofstream outputFile(output_file, std::ios::binary);
    if (outputFile.is_open()) {
        int len = vec.size();
        for (int i = 0 ; i < len; i++) {
            outputFile.write(reinterpret_cast<const char*>(&vec[i]), sizeof(vec[i]));            
        }
        outputFile.close();
        std::cout << output_file << " 写入完成。" << std::endl;
    } else {
        std::cout << output_file << " 无法打开输出文件。" << std::endl;
    }
}

void readCSRFile(std::string output_file,std::vector<int>& vec,int len) {
    FILE * fp = fopen64(output_file.c_str(),"r");
    assert(fp!=NULL);
    uint rd = 0;
    for(uint i=0;i<len;i++) {
        int s;    
        rd += fread(&s, sizeof(int), 1, fp); 
        vec[i] = s;
    }
}

void printVec(std::vector<int>& vec) {
    int len = vec.size();
    for (int i = 0 ; i < len ; i++) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}

void COO2CSR(const std::string& csvFile, std::vector<int>& src, std::vector<int>& dstRange) {
    std::ifstream file(csvFile);
    std::map<int,std::vector<int>> data;
    int startNode, endNode;
    char dot;
    int numEdges = 0;
    while (file >> startNode >> dot >> endNode) {
        data[endNode].push_back(startNode);
        numEdges++;
    }
    file.close();

    int numNodes = 9;
    src.resize(numEdges);
    dstRange.resize(numNodes + 1);

    int idx = 0;
    std::vector<int> tmp;
    for (int i = 0; i < numNodes; ++i) {
        if (data.find(i) != data.end()){
            tmp = data[i];
            sort(tmp.begin(),tmp.end());
            dstRange[i] = idx;
            for(int index = 0 ; index < tmp.size() ; index++){
                src[idx++] = tmp[index];
            }
        } else {
            dstRange[i] = idx;
        }
    }
    dstRange[numNodes] = numEdges;
}

void edgesConvert2CSR(const std::string& inputFilename, const std::string& savePath,std::vector<int>& src, std::vector<int>& dstRange) {
    std::ifstream inputFile(inputFilename);
    std::string line;
    std::getline(inputFile, line);
    std::stringstream confss(line);
    std::vector<int> confData; // [nodeNUM,edgeNUM]
    std::string token;
    while (std::getline(confss, token, ',')) { 
        int value = std::stoi(token);
        confData.push_back(value);
    }
    std::map<int,std::vector<int>> graph;

    if (inputFile) {   
        while (std::getline(inputFile, line)) {
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
        inputFile.close();
        std::cout << "read txt file success." << std::endl;
    } else {
        std::cerr << "Failed to open input or output file." << std::endl;
    }

    src.resize(confData[1]);
    dstRange.resize(confData[0]+1);
    int saveIndex = 0;
    for(int idx = 0 ; idx < confData[0] ; idx++) {
        if (graph.find(idx) != graph.end()){
            sort(graph[idx].begin(),graph[idx].end());
            dstRange[idx] = saveIndex;
            for(int index = 0 ; index < graph[idx].size() ; index++){
                src[saveIndex++] = graph[idx][index];
            }
        } else {
            dstRange[idx] = saveIndex;
        }
    }
    dstRange[confData[0]] = saveIndex;
    std::string srcPath = savePath+"/srcList.bin";
    std::string rangePath = savePath+"/range.bin";
    writeBinFile(srcPath,src);
    writeBinFile(rangePath,dstRange);

}




int main(int argc, char* argv[]) {
    std::vector<int> src;
    std::vector<int> dstRange;
    std::string txtFile = argv[1];
    std::string savePath = argv[2];
    edgesConvert2CSR(txtFile,savePath,src, dstRange);
    return 0;
}
