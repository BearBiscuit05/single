#include <cuda_runtime.h>
#include "sample.h"
#include <cassert>
#include <iostream>



template <int BLOCK_SIZE, int TILE_SIZE>
__global__ void sample_hop_kernel(
    int* graphEdge,int* bound,int* seed,
    int seed_num,int fanout,
    int* out_src,int* out_dst,
    unsigned long random_states,int gapNUM) {
    // assert(BLOCK_SIZE == blockDim.x);
    fanout = fanout - 1;
    const size_t block_start = TILE_SIZE * blockIdx.x;
    const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   
    
    curandStateXORWOW_t local_state;
    curand_init(random_states+idx,0,0,&local_state);
    
    for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
        int rid = seed[index];
        // if (index < seed_num)
        //     printf("node id: %d\n",rid);
        if (index < seed_num && rid >= 0) {
            int off = bound[rid*2] + gapNUM;
            int len = bound[rid*2+1] - bound[rid*2] - gapNUM;
            out_src[index] = rid;
            out_dst[index] = rid;
            // printf("node gap: %d\n",gapNUM);
            // printf("node off: %d\n",off);
            // printf("neri len: %d\n",len);
            if (len <= fanout) {
                size_t j = 0;
                for (; j < len; ++j) {
                    out_dst[seed_num + index * fanout + j] = rid;
                    out_src[seed_num + index * fanout + j] = graphEdge[off + j];
                }
                for (; j < fanout; ++j) {
                    out_dst[seed_num + index * fanout + j] = -1;
                    out_src[seed_num + index * fanout + j] = -1;
                }
            } else {
                for (int j = 0; j < fanout; j++) {
                    int selected_j = curand(&local_state) % (len - j);
                    int selected_node_id = graphEdge[off + selected_j];
                    out_dst[seed_num + index * fanout + j] = rid;
                    out_src[seed_num + index * fanout + j] = selected_node_id;
                    graphEdge[off + selected_j] = graphEdge[off+len-j-1];
                    graphEdge[off+len-j-1] = selected_node_id;
                }
            } 
        }
    }
    // random_states[idx] = local_state;
}   


template <int BLOCK_SIZE, int TILE_SIZE>
__global__ void graph_halo_merge_kernel(
    int* edge,int* bound,
    int* halos,int* halo_bound,
    int nodeNUM,unsigned long random_states
) {
    const size_t block_start = TILE_SIZE * blockIdx.x;
    const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   
    curandStateXORWOW_t local_state;
    curand_init(random_states+idx,0,0,&local_state);
    for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
        if (index < nodeNUM) {
            int rid = index;
            int startptr = bound[index*2+1];
            int endptr = bound[index*2+2];
            int space = endptr - startptr;
            int off = halo_bound[rid];
            int len = halo_bound[rid+1] - halo_bound[rid];
            if (len > 0) {
                // 存在可以补充的位置
                if (space < len) {
                    // 可补充边大于预留位置
                    for (int j = 0; j < space; j++) {
                        int selected_j = curand(&local_state) % (len - j);
                        int selected_id = halos[off + selected_j];
                        edge[startptr++] = selected_id;
                        halos[off + selected_j] = halos[off+len-j-1];
                        halos[off+len-j-1] = selected_id;
                    }
                    bound[index*2+1] = startptr;
                } else {
                    // 可补充边小于预留位置
                    for (int j = 0; j < len; j++) {
                        edge[startptr++] = halos[off + j];
                    }
                    bound[index*2+1] = startptr;
                }
            }
        }
    }
}


inline int RoundUpDiv(int target, int unit) {
  return (target + unit - 1) / unit;
}


using StreamHandle = void*;

void sample_hop(
    int* graphEdge,int* bound,int* seed,
    int seed_num,int fanout,int* out_src,
    int* out_dst,int gapNUM)
{   

    const int slice = 1024;
    const int blockSize = 256;
    int steps = RoundUpDiv(seed_num,slice);
    
    dim3 grid(steps);
    dim3 block(blockSize);
    // printf("grids:%d \n",steps);
    // printf("block:%d \n",blockSize);
    unsigned long timeseed =
        std::chrono::system_clock::now().time_since_epoch().count();
    sample_hop_kernel<blockSize, slice>
    <<<grid,block>>>(graphEdge,bound,seed,
    seed_num,fanout,out_src,out_dst,timeseed,gapNUM);
    // printf("TESTING.....\n");
    cudaDeviceSynchronize();
}


void graph_halo_merge(
    int* edge,int* bound,
    int* halos,int* halo_bound,int nodeNUM) {
    
    const int slice = 1024;
    const int blockSize = 256;
    int steps = RoundUpDiv(nodeNUM,slice);

    dim3 grid(steps);
    dim3 block(blockSize);
    unsigned long timeseed =
        std::chrono::system_clock::now().time_since_epoch().count();
    graph_halo_merge_kernel<blockSize, slice>
    <<<grid,block>>>(edge,bound,halos,halo_bound,nodeNUM,timeseed);
    cudaDeviceSynchronize();
    
}