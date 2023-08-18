#ifndef SAMPLE_H
#define SAMPLE_H

#include "common.cuh"

void sample_hop(
    int* graphEdge,int* bound,int* seed,
    int seed_num,int fanout,int* out_src,
    int* out_dst,size_t* num_out);

void graph_halo_merge(
    int* edge,int* bound,
    int* halos,int* halo_bound,int nodeNUM);

void graph_mapping(
    int* nodeList,int* nodeSRC,
    int* nodeDST,int* newNodeSRC,
    int* newNodeDST,int* uniqueList,
    int edgeNUM,int uniqueNUM
);



#endif