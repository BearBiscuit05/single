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
    std::string inputGraphPath;
    int clusterBSize;

    GlobalConfig() {};
    GlobalConfig(std::string filepath);
    int getMaxClusterVolume();
    double getAverageDegree();
};

#endif // GLOBALCONFIG_H
