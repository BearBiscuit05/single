#include "common.cuh"




template <int BLOCK_SIZE, int TILE_SIZE>
__global__ void sample_hop_kernel(
    int* graphEdge,int* bound,int* seed,
    int seed_num,int fanout,
    int* out_src,int* out_dst,
    unsigned long random_states) {
    assert(BLOCK_SIZE == blockDim.x);
    const size_t block_start = TILE_SIZE * blockIdx.x;
    const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   
    
    curandStateXORWOW_t local_state;
    curand_init(random_states+idx,0,0,&local_state);
    
    for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
        int rid = seed[index];
        if (index < seed_num && rid >= 0) {
            int off = bound[rid*2];
            int len = bound[rid*2+1] - bound[rid*2];
            if (len <= fanout) {
                size_t j = 0;
                for (; j < len; ++j) {
                    out_dst[index * fanout + j] = rid;
                    out_src[index * fanout + j] = graphEdge[off + j];
                }
                for (; j < fanout; ++j) {
                    out_dst[index * fanout + j] = -1;
                    out_src[index * fanout + j] = -1;
                }
            } else {
                for (int j = 0; j < fanout; ++j) {
                    int selected_j = curand(&local_state) % (len - j);
                    int selected_node_id = graphEdge[off + selected_j];
                    out_dst[index * fanout + j] = rid;
                    out_src[index * fanout + j] = selected_node_id;
                    graphEdge[off + selected_j] = graphEdge[off+len-j-1];
                    graphEdge[off+len-j-1] = selected_node_id;
                }
            } 
        }
    }
}   


template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void sample_count_edge(int *edge_src, size_t *item_prefix,
                           const size_t num_input, const size_t fanout) {
  assert(BLOCK_SIZE == blockDim.x);
  using BlockReduce = typename cub::BlockReduce<size_t, BLOCK_SIZE>;
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  size_t count = 0;
#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_input) {
      for (size_t j = 0; j < fanout; j++) {
        if (edge_src[index * fanout + j] != -1) {
          ++count;
        }
      }
    }
  }

  __shared__ typename BlockReduce::TempStorage temp_space;

  count = BlockReduce(temp_space).Sum(count);

  if (threadIdx.x == 0) {
    item_prefix[blockIdx.x] = count;
    if (blockIdx.x == 0) {
      item_prefix[gridDim.x] = 0;
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void sample_compact_edge(const int *tmp_src, const int *tmp_dst,
                             int *out_src, int *out_dst, size_t *num_out,
                             const size_t *item_prefix, const int num_input,
                             const int fanout) {
  assert(BLOCK_SIZE == blockDim.x);

  using BlockScan = typename cub::BlockScan<size_t, BLOCK_SIZE>;

  constexpr const size_t VALS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;

  __shared__ typename BlockScan::TempStorage temp_space;

  const size_t offset = item_prefix[blockIdx.x];

  BlockPrefixCallbackOp<size_t> prefix_op(0);

  // count successful placements
  for (int i = 0; i < VALS_PER_THREAD; ++i) {
    const size_t index = threadIdx.x + i * BLOCK_SIZE + blockIdx.x * TILE_SIZE;

    size_t item_per_thread = 0;
    if (index < num_input) {
      // printf("index : %d \n",index);
      for (size_t j = 0; j < fanout; j++) {
        if (tmp_src[index * fanout + j] != -1) {
          item_per_thread++;
        }
      }
      // printf("item_per_thread : %d \n",item_per_thread);
    }

    size_t item_prefix_per_thread = item_per_thread;
    BlockScan(temp_space)
        .ExclusiveSum(item_prefix_per_thread, item_prefix_per_thread,
                      prefix_op);
    __syncthreads();
    
    for (size_t j = 0; j < item_per_thread; j++) {
      // printf("item_prefix_per_thread : %d \n",item_prefix_per_thread);
      // printf("j : %d \n",j);
      out_src[offset + item_prefix_per_thread + j] =
          tmp_src[index * fanout + j];
      out_dst[offset + item_prefix_per_thread + j] =
          tmp_dst[index * fanout + j];
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // printf("item_prefix[gridDim.x] %d \n",item_prefix[gridDim.x]);
    *num_out = item_prefix[gridDim.x];
  }
}

template <int BLOCK_SIZE, int TILE_SIZE>
__global__ void graph_halo_merge_kernel(
    int* edge,int* bound,
    int* halos,int* halo_bound,int nodeNUM,
    int gap,unsigned long random_states
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
            int off = halo_bound[rid] + gap;
            int len = halo_bound[rid+1] - off;
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
    int* out_dst,size_t* num_out)
{   
    const int slice = 1024;
    const int blockSize = 256;
    int steps = RoundUpDiv(seed_num,slice);
    
    dim3 grid(steps);
    dim3 block(blockSize);
    unsigned long timeseed =
        std::chrono::system_clock::now().time_since_epoch().count();
    
    int *tmp_src = static_cast<int *>(AllocDataSpace(sizeof(int) * seed_num * fanout));
    int *tmp_dst = static_cast<int *>(AllocDataSpace(sizeof(int) * seed_num * fanout));

    sample_hop_kernel<blockSize, slice>
        <<<grid,block>>>(graphEdge,bound,seed,
        seed_num,fanout,tmp_src,tmp_dst,timeseed);
    CUDA_CALL(cudaDeviceSynchronize());
    
    size_t *item_prefix = static_cast<size_t *>(AllocDataSpace(sizeof(size_t) * (grid.x + 1)));
    std::vector<size_t> dev_item(grid.x  + 1,0);
	  cudaMemcpy(item_prefix, dev_item.data(), sizeof(size_t)*(grid.x  + 1), cudaMemcpyHostToDevice);

    sample_count_edge<blockSize, slice>
      <<<grid, block>>>(tmp_src, item_prefix, seed_num, fanout);
    CUDA_CALL(cudaDeviceSynchronize());

    cudaMemcpy(dev_item.data(), item_prefix, sizeof(size_t)*(grid.x  + 1), cudaMemcpyDeviceToHost);


    size_t workspace_bytes;
    CUDA_CALL(cub::DeviceScan::ExclusiveSum(
        nullptr, workspace_bytes, static_cast<size_t *>(nullptr),
        static_cast<size_t *>(nullptr), grid.x + 1));
    CUDA_CALL(cudaDeviceSynchronize());

    void *workspace = AllocDataSpace(workspace_bytes);
    CUDA_CALL(cub::DeviceScan::ExclusiveSum(workspace, workspace_bytes,
                                            item_prefix, item_prefix, grid.x + 1));
    CUDA_CALL(cudaDeviceSynchronize());
    cudaMemcpy(dev_item.data(), item_prefix, sizeof(size_t)*(grid.x  + 1), cudaMemcpyDeviceToHost);

    sample_compact_edge<blockSize, slice>
      <<<grid, block>>>(tmp_src, tmp_dst, out_src, out_dst,
                                      num_out, item_prefix, seed_num, fanout);
    CUDA_CALL(cudaDeviceSynchronize());

    FreeDataSpace(workspace);
    FreeDataSpace(item_prefix);
    FreeDataSpace(tmp_src);
    FreeDataSpace(tmp_dst);

}


void graph_halo_merge(
    int* edge,int* bound,
    int* halos,int* halo_bound,int nodeNUM,int gap) {
    
    const int slice = 1024;
    const int blockSize = 256;
    int steps = RoundUpDiv(nodeNUM,slice);

    dim3 grid(steps);
    dim3 block(blockSize);
    unsigned long timeseed =
        std::chrono::system_clock::now().time_since_epoch().count();
    graph_halo_merge_kernel<blockSize, slice>
    <<<grid,block>>>(edge,bound,halos,halo_bound,nodeNUM,gap,timeseed);
    cudaDeviceSynchronize();
    
}

// void graph_mapping(
//     int* nodeList,int* mappingTable,int nodeNUM,int mappingNUM
// ) {
//     const int slice = 1024;
//     const int blockSize = 256;
//     int steps = RoundUpDiv(nodeNUM,slice);
//     dim3 grid(steps);
//     dim3 block(blockSize);
//     unsigned long timeseed =
//         std::chrono::system_clock::now().time_since_epoch().count();
//     graph_mapping_kernel<blockSize, slice>
//     <<<grid,block>>>(nodeList,mappingTable,nodeNUM,mappingNUM);
//     cudaDeviceSynchronize();
// }



