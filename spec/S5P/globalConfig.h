#ifndef GLOBALCONFIG_H
#define GLOBALCONFIG_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>

class GlobalConfig {
private:
    std::map<std::string, std::string> properties;

public:
    int k;
    int hashNum;
    float alpha;
    double beta;
    double tao;
    int batchSize;
    int threads;
    int partitionNum;
    int vCount;
    int eCount;
    int eCount_B;
    int eCount_S;
    std::string inputGraphPath;
    std::string inputGraphPath_B;
    std::string inputGraphPath_S;
    std::string outputGraphPath;
    std::string outputResultPath;
    
    GlobalConfig();
    GlobalConfig(std::string filepath);
    int clusterBSize;
    int getMaxClusterVolume();
    double getAverageDegree();
};

#endif // GLOBALCONFIG_H
