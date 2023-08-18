#include "common.cuh"

void *AllocDataSpace(size_t nbytes) {
    void *ret = nullptr;
    CUDA_CALL(cudaMalloc(&ret, nbytes));
    return ret;
}

void FreeDataSpace(void *ret) {
  CUDA_CALL(cudaFree(ret));
}
