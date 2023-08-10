#ifndef SAMGRAPH_RANDOM_STATES_H
#define SAMGRAPH_RANDOM_STATES_H
#include <iostream>
#include <curand_kernel.h>
#include <vector>
#include <cassert>
#include <chrono>
#include <numeric>
// #define CHECK(x) \
//   if (!(x))      \
//   LogMessageFatal(__FILE__, __LINE__) << "Check failed: " #x << ' '

// #define CUDA_CALL(func)                                      \
//   {                                                          \
//     cudaError_t e = (func);                                  \
//     CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading) \
//         << "CUDA: " << cudaGetErrorString(e);                \
//   }

class GPURandomStates {
 public:
    GPURandomStates(int num_states);
    curandState* GetStates() { return _states.data(); };
    int NumStates() { return _num_states; };

    private:
        std::vector<curandState> _states;
        int _num_states;
};

#endif
