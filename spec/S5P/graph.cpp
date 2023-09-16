#include "graph.h"
#include <iostream>


Graph::Graph() {}

Graph::Graph(GlobalConfig config) : fileStream(config.inputGraphPath) {
    this->vCount = config.vCount;
    this->eCount = config.eCount;
    this->graphpath = config.inputGraphPath;
}

Graph::~Graph() {
    clear();
}

Graph::Graph(const Graph& other) : vCount(other.vCount), eCount(other.eCount), graphpath(std::move(other.graphpath)) {}


Graph& Graph::operator=(const Graph& other) {
    if (this != &other) {
        vCount = other.vCount;
        eCount = other.eCount;
        graphpath = other.graphpath;
    }
    return *this;
}

int Graph::readStep(Edge& edge) {
    std::string line;
    if (std::getline(this->fileStream, line)) {
        if (line.empty() || line[0] == '#')
            return readStep(edge);

        size_t tabPos = line.find('\t');
        int srcVId = std::stoi(line.substr(0, tabPos));
        int destVId = std::stoi(line.substr(tabPos + 1));

        if (srcVId == destVId)
            return readStep(edge);
        edge.srcVId = srcVId;
        edge.destVId = destVId;
        return 0;
    }
    std::cout << "read end..." << std::endl;
    return -1; // Return an empty edge if end of file is reached
}

void Graph::readGraphFromFile() {
    fileStream.seekg(0, std::ios::beg);
    if (!fileStream.is_open()) {
        std::cerr << "Error: Unable to open the graph file." << std::endl;
        return;
    }
}

