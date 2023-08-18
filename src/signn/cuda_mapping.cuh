#include "cuda_hashtable.cuh"

template <typename IdType>
void GPUMapEdges( IdType * global_src, IdType * new_global_src,
                  IdType * global_dst, IdType * new_global_dst,
                  size_t num_edges, DeviceOrderedHashTable<IdType> table
                );