#ifndef SAMPLE_H
#define SAMPLE_H

#include "common.cuh"
#include "cuda_hashtable.cuh"

void sample_hop(
    int* graphEdge,int* bound,int* seed,
    int seed_num,int fanout,int* out_src,
    int* out_dst,int gapNUM);

void graph_halo_merge(
    int* edge,int* bound,
    int* halos,int* halo_bound,int nodeNUM);

void graph_mapping(
    int* nodeList,int* mappingTable,int nodeNUM,int mappingNUM
);

void mutiLayersSample(
    int* graphEdge,int* bound,
    int* seed,int64_t seed_num,int* fanouts,int64_t fanoutNUM,
    int* outSrcNodes,int* outDstNodes,int* outList,
    int* outrawnodesid,int64_t outnodesNUM
);

#endif