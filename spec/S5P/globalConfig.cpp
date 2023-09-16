#include "globalConfig.h"

GlobalConfig::GlobalConfig(){}

GlobalConfig::GlobalConfig(std::string filepath) {
    std::ifstream configFile(filepath);
    std::string line;
    while (std::getline(configFile, line)) {
        // Skip empty lines and lines starting with '#'
        if (line.empty() || line[0] == '#')
            continue;

        // Find the position of '=' to separate the key and value
        size_t delimiterPos = line.find('=');
        if (delimiterPos == std::string::npos) {
            std::cerr << "Error: Invalid line format in the configuration file: " << line << std::endl;
            continue;
        }

        std::string key = line.substr(0, delimiterPos);
        std::string value = line.substr(delimiterPos + 1);
        std::cout << key  << " = " << value << std::endl;
        properties[key] = value;
    }

    hashNum = std::stoi(properties["hashNum"]);
    alpha = std::stof(properties["alpha"]);
    beta = std::stod(properties["beta"]);
    tao = std::stod(properties["tao"]);
    batchSize = std::stoi(properties["batchSize"]);
    threads = std::stoi(properties["threads"]);
    partitionNum = std::stoi(properties["partitionNum"]);
    vCount = std::stoi(properties["vCount"]);
    eCount = std::stoi(properties["eCount"]);
    eCount_B = std::stoi(properties["eCount_B"]);
    k = std::stoi(properties["k"]);
    eCount_S = std::stoi(properties["eCount_S"]);
    inputGraphPath = properties["inputGraphPath"];
    inputGraphPath_B = properties["inputGraphPath_B"];
    inputGraphPath_S = properties["inputGraphPath_S"];
    outputGraphPath = properties["outputGraphPath"];
    outputResultPath = properties["outputGraphPath"];
}

int GlobalConfig::getMaxClusterVolume() {
    return  eCount / k;
}

double GlobalConfig::getAverageDegree() {
    return 2.0 * eCount / vCount;
}