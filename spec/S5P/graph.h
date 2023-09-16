#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <iostream>
#include "globalConfig.h"

// Define the Edge class
class Edge {
public:
    int srcVId;
    int destVId;
    Edge() {};
    Edge(int srcVId, int destVId, int weight) : srcVId(srcVId), destVId(destVId) {};
    bool operator!=(const Edge& other) const {
        return (srcVId != other.srcVId) && (destVId != other.destVId);
    }
};

// Define the Graph class
class Graph {
public:
    int vCount;
    int eCount;
    std::ifstream fileStream;
    std::string graphpath;


    Graph();
    Graph(GlobalConfig config);
    ~Graph();
    Graph(const Graph& other);
    Graph& operator=(const Graph& other);
    int readStep(Edge& edge);
    void readGraphFromFile();
    void addEdge(int srcVId, int destVId);
    int getVCount();
    int getECount();
    void clear();
};

#endif // GRAPH_H
