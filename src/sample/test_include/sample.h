#ifndef SAMPLE_H
#define SAMPLE_H

#include "random_state.h"

void sample_hop(
    int* graphEdge,int* bound,int* seed,
    int seed_num,int fanout,int* out_src,
    int* out_dst,int gapNUM);

void graph_halo_merge(
    int* edge,int* bound,
    int* halos,int* halo_bound,int nodeNUM);

//void test_random();

#endif