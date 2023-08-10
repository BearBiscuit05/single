#include <cassert>
#include <chrono>
#include <numeric>

#include "random_state.h"


__global__ void init_random_states(curandState *states, size_t num,
                                   unsigned long seed) {
  size_t threadId = threadIdx.x + blockIdx.x * blockDim.x;
  if (threadId < num) {
    curand_init(seed+threadId, 0, 0, &states[threadId]);
  }
}


GPURandomStates::GPURandomStates(
    int num_states) {
     _num_states = num_states;

    _states.resize(_num_states);

    const int BlockSize = 256;
    const dim3 grid((_num_states + BlockSize - 1) / BlockSize);
    const dim3 block(BlockSize);

    unsigned long seed =
        std::chrono::system_clock::now().time_since_epoch().count();
    init_random_states<<<grid, block>>>(_states.data(), _num_states, seed);
    std::cout << "get in random..." << std::endl;
    cudaDeviceSynchronize();
}

