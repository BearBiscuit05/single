#include <cuda_runtime.h>
#include "sample.h"
#include <cassert>
#include <iostream>



template <int BLOCK_SIZE, int TILE_SIZE>
__global__ void sample_2hop_kernel(
    int* bound,int* graphEdge,int* seed,
    int seed_num,int fanout,
    int* out_src,int* out_dst,unsigned long random_states) {
    
    assert(BLOCK_SIZE == blockDim.x);
    
    const size_t block_start = TILE_SIZE * blockIdx.x;
    const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   

    curandStateXORWOW_t local_state;
    // curandState local_state = random_states[idx];
    curand_init(random_states+idx,0,0,&local_state);

    for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
        
        if (index < seed_num) {
            int rid = seed[index];
            int off = bound[rid*2];
            int len = bound[rid*2+1] - bound[rid*2];
            if (len <= fanout) {
                size_t j = 0;
                for (; j < len; ++j) {
                    out_src[index * fanout + j] = rid;
                    out_dst[index * fanout + j] = graphEdge[off + j];
                }

                // for (; j < fanout; ++j) {
                //     out_src[index * fanout + j] = -3;
                //     out_dst[index * fanout + j] = -3;
                // }
            } else {
                for (int j = 0; j < fanout; j++) {
                    // int selected_j = curand(&local_state) % (len - j);
                    int selected_j = curand(&local_state) % (len);
                    // int selected_j = j;
                    int selected_node_id = graphEdge[off + selected_j];
                    out_src[index * fanout + j] = rid;
                    out_dst[index * fanout + j] = selected_node_id;
                    //indices[off + selected_j] = indices[off+len-j-1];
                    //indices[off+len-j-1] = selected_node_id;
                }
            } 
        }
    }
    // random_states[idx] = local_state;
}   


template <int BLOCK_SIZE, int TILE_SIZE>
__global__ void graph_halo_merge_kernel(
    int* edge,int* bound,
    int* halos,int* halo_bound,int nodeNUM,unsigned long random_states
) {
    const size_t block_start = TILE_SIZE * blockIdx.x;
    const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   

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
                        edge[startptr++] = halos[off + j];
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

void sample_2hop(
    int* bound,int* graphEdge,int* seed,
    int seed_num,int fanout,int* out_src,
    int* out_dst)
{   

    const int slice = 1024;
    const int blockSize = 256;
    // int batchsize = 1024;
    int steps = RoundUpDiv(seed_num,slice);
    //GPURandomStates randomStates(steps*blockSize);
    
    dim3 grid(steps);
    dim3 block(blockSize);

    // cudaStream_t stream;
    // StreamHandle stream; 
    //CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    // auto stream_copy = static_cast<StreamHandle>(stream);
    // auto cu_stream = static_cast<cudaStream_t>(stream);
    
    // std::cout << "============hello world=================" << std::endl;
    unsigned long timeseed =
        std::chrono::system_clock::now().time_since_epoch().count();
    sample_2hop_kernel<blockSize, slice>
    <<<grid,block>>>(bound,graphEdge,seed,
    seed_num,fanout,out_src,out_dst,timeseed);
    cudaDeviceSynchronize();
    // CUDA_CALL(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
    // cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
}


void graph_halo_merge(
    int* edge,int* bound,
    int* halos,int* halo_bound,int nodeNUM) {
    
    const int slice = 1024;
    const int blockSize = 256;
    // int batchsize = 1024;
    int steps = RoundUpDiv(nodeNUM,slice);

    dim3 grid(steps);
    dim3 block(blockSize);
    unsigned long timeseed =
        std::chrono::system_clock::now().time_since_epoch().count();
    graph_halo_merge_kernel<blockSize, slice>
    <<<grid,block>>>(edge,bound,halos,halo_bound,nodeNUM,timeseed);
    cudaDeviceSynchronize();
}