#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <map>
#include <sstream>
#include <cassert>
#include <torch/extension.h>
#include "sample.h"


void torch_sample_2hop(
    torch::Tensor &bound,torch::Tensor &graphEdge,
    torch::Tensor &seed,int seed_num,int fanout,
    torch::Tensor &out_src,torch::Tensor &out_dst
    ) {
    sample_2hop(
        (int*) bound.data_ptr(),(int*) graphEdge.data_ptr(),(int*) seed.data_ptr(),
        seed_num,fanout,(int*) out_src.data_ptr(),
        (int*) out_dst.data_ptr());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_sample_2hop",
          &torch_sample_2hop,
          "2 hop sample");
}

// TORCH_LIBRARY(sample_hop_new, m) {
//     m.def("torch_sample_2hop", torch_sample_2hop);
// }
