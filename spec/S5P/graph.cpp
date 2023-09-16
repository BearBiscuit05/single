#include "graph.h"
#include <iostream>

// Edge class
Edge::Edge() {}

Edge::Edge(int srcVId, int destVId, int weight) : srcVId(srcVId), destVId(destVId), weight(weight) {}

int Edge::getSrcVId() const {
    return this->srcVId;
}

int Edge::getDestVId() const {
    return this->destVId;
}

int Edge::getWeight() const {
    return this->weight;
}

void Edge::addWeight() {
    this->weight++;
}

// Graph class
Graph::Graph() {}

Graph::Graph(GlobalConfig config) : fileStream(config.inputGraphPath) {
    this->edgeList = std::vector<Edge>();
    this->vCount = config.vCount;
    this->eCount = config.eCount;
    this->graphpath = config.inputGraphPath;
}

Graph::~Graph() {
    clear();
}

Graph::Graph(const Graph& other) : vCount(other.vCount), eCount(other.eCount), graphpath(std::move(other.graphpath)) {
    edgeList = std::move(other.edgeList);
}


Graph& Graph::operator=(const Graph& other) {
    if (this != &other) {
        vCount = other.vCount;
        eCount = other.eCount;
        graphpath = other.graphpath;
        edgeList = other.edgeList;
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
        edge.weight = 1;
        return 0;
    }
    std::cout << "read end..." << std::endl;
    return -1; // Return an empty edge if end of file is reached
}

void Graph::readGraphFromFile() {
    // // this->fileStream.open(graphpath);
    // std::cout << "begin..." << std::endl;
    // fileStream.clear();
    // std::cout << "clear..." << std::endl;
    // std::cout << &fileStream << std::endl;
    fileStream.seekg(0, std::ios::beg);
    // std::cout << "seekg..." << std::endl;
    if (!fileStream.is_open()) {
        std::cerr << "Error: Unable to open the graph file." << std::endl;
        return;
    }
}

void Graph::addEdge(int srcVId, int destVId) {
    this->edgeList.emplace_back(srcVId, destVId, 1);
}

std::vector<Edge> Graph::getEdgeList() {
    return this->edgeList;
}

int Graph::getVCount() {
    return this->vCount;
}

int Graph::getECount() {
    return this->eCount;
}

void Graph::clear() {
    edgeList.clear();
    vCount = 0;
    eCount = 0;
    graphpath.clear();
}
