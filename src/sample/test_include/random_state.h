#ifndef SAMGRAPH_RANDOM_STATES_H
#define SAMGRAPH_RANDOM_STATES_H

#include <curand_kernel.h>
#include <vector>



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
