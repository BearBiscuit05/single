#ifndef COMMON_H
#define COMMON_H

#pragma once
//#include <torch/extension.h>
#include <iostream>
#include <curand_kernel.h>
#include <vector>
#include <chrono>
#include <numeric>
#include <fstream>
#include <algorithm>
#include <map>
#include <sstream>
#include <cassert>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cub/cub.cuh>




#define SLICE 1024
#define BLOCKSIZE 256


#define CUDA_CALL(func)                                      \
  {                                                          \
    cudaError_t e = (func);                                  \
    if(e!=cudaSuccess) { \
		std::cout << "Cuda failure " << __FILE__ << ":" << __LINE__ << ", " << cudaGetErrorString(e) << " (" << e << ")" << std::endl; \
		exit(0); \
	 }\
  }


inline __device__ int64_t
AtomicCAS(int64_t* const address, const int64_t compare, const int64_t val) {
  using Type = unsigned long long int;  // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return atomicCAS(
      reinterpret_cast<Type*>(address), static_cast<Type>(compare),
      static_cast<Type>(val));
}

inline __device__ int32_t
AtomicCAS(int32_t* const address, const int32_t compare, const int32_t val) {
  using Type = int;  // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return atomicCAS(
      reinterpret_cast<Type*>(address), static_cast<Type>(compare),
      static_cast<Type>(val));
}

void* AllocDataSpace(size_t nbytes);
void FreeDataSpace(void* ptr);

template <typename IdType>
struct BlockPrefixCallbackOp {
  IdType running_total_;

  __device__ BlockPrefixCallbackOp(const IdType running_total)
      : running_total_(running_total) {}

  __device__ IdType operator()(const IdType block_aggregate) {
    const IdType old_prefix = running_total_;
    running_total_ += block_aggregate;
    return old_prefix;
  }
};

#endif