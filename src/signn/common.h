#ifndef COMMON_H
#define COMMON_H


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
#include <torch/extension.h>

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

#endif