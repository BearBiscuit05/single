#ifndef SAMPLE_H
#define SAMPLE_H

#include "random_state.h"

void sample_2hop(
    int* bound,int* graphEdge,int* seed,
    int seed_num,int fanout,int* out_src,
    int* out_dst);

//void test_random();

#endif