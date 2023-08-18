#include <cassert>
#include <cstdio>
#include "cuda_mapping.cuh"

inline int RoundUpDiv(int target, int unit) {
  return (target + unit - 1) / unit;
}



template <typename IdType,size_t BLOCK_SIZE, size_t TILE_SIZE>
__device__ void map_node_ids(const IdType *const global,
                             IdType *const new_global, const size_t num_input,
                             const DeviceOrderedHashTable<IdType> &table) {
  assert(BLOCK_SIZE == blockDim.x);

  using Bucket = typename OrderedHashTable<IdType>::Mapping;

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = min(TILE_SIZE * (blockIdx.x + 1), num_input);

  for (size_t idx = threadIdx.x + block_start; idx < block_end;
       idx += BLOCK_SIZE) {
    const Bucket &bucket = *table.Search(global[idx]);
    new_global[idx] = bucket.local;
  }
}

template <typename IdType,size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void map_edge_ids(const IdType *const global_src,
                             IdType *const new_global_src,
                             const IdType *const global_dst,
                             IdType *const new_global_dst,
                             const size_t num_edges,
                             DeviceOrderedHashTable<IdType> table) {
  assert(BLOCK_SIZE == blockDim.x);
  assert(2 == gridDim.y);

  if (blockIdx.y == 0) {
    map_node_ids<IdType,BLOCK_SIZE, TILE_SIZE>(global_src, new_global_src, num_edges,
                                        table);
  } else {
    map_node_ids<IdType,BLOCK_SIZE, TILE_SIZE>(global_dst, new_global_dst, num_edges,
                                        table);
  }
}

template <typename IdType>
void GPUMapEdges( IdType * global_src, IdType * new_global_src,
                  IdType * global_dst, IdType * new_global_dst,
                  size_t num_edges, DeviceOrderedHashTable<IdType> table
                ) {
  const int slice = 1024;
  const int blockSize = 256;
  const size_t num_tiles = RoundUpDiv(num_edges,slice);
  const dim3 grid(num_tiles, 2);
  const dim3 block(blockSize);

  map_edge_ids<IdType,blockSize, slice>
      <<<grid, block>>>(global_src, new_global_src, global_dst,
                                      new_global_dst, num_edges, table);
}

template void GPUMapEdges<int32_t>(
    int32_t * global_src, int32_t * new_global_src,
    int32_t * global_dst, int32_t * new_global_dst,
    size_t num_edges, DeviceOrderedHashTable<int32_t> table
);

template void GPUMapEdges<int64_t>(
    int64_t * global_src, int64_t * new_global_src,
    int64_t * global_dst, int64_t * new_global_dst,
    size_t num_edges, DeviceOrderedHashTable<int64_t> table
);
