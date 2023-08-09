#include <torch/extension.h>
#include "sample_node.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <map>
#include <sstream>
#include <cassert>
#include <pybind11/pybind11.h>

void torch_launch_sample_full(torch::Tensor &outputSRC1,
                        torch::Tensor &outputDST1,
                       const torch::Tensor &graphEdge,
                       const torch::Tensor &boundList,
                       const torch::Tensor &trainNode,
                       int64_t nodeNUM,
                       const int64_t gpuDeviceIndex) {
    launch_sample_full((int*) outputSRC1.data_ptr(),
                 (int*) outputDST1.data_ptr(),
                 (const int*) graphEdge.data_ptr(),
                 (const int*) boundList.data_ptr(),
                 (const int*) trainNode.data_ptr(),
                 nodeNUM,gpuDeviceIndex) ;
}

void torch_launch_sample_1hop(torch::Tensor &outputSRC1,
                        torch::Tensor &outputDST1,
                       const torch::Tensor &graphEdge,
                       const torch::Tensor &boundList,
                       const torch::Tensor &trainNode,
                       int64_t sampleNUM1,
                       int64_t nodeNUM,
                       const int64_t gpuDeviceIndex) {
    launch_sample_1hop(static_cast<int*>(outputSRC1.data_ptr()),
                    static_cast<int*>(outputDST1.data_ptr()), 
                    (const int *)graphEdge.data_ptr(),
                    (const int *)boundList.data_ptr(),
                    (const int *)trainNode.data_ptr(),
                    sampleNUM1,
                    nodeNUM,gpuDeviceIndex);
}


void torch_launch_sample_2hop(torch::Tensor &outputSRC1,
                              torch::Tensor &outputDST1,
                              torch::Tensor &outputSRC2,
                              torch::Tensor &outputDST2,
                       const torch::Tensor &graphEdge,
                       const torch::Tensor &boundList,
                       const torch::Tensor &trainNode,
                       int64_t sampleNUM1,
                       int64_t sampleNUM2,
                       int64_t nodeNUM,
                       const int64_t gpuDeviceIndex) {
    //auto t_beg = std::chrono::high_resolution_clock::now();
    launch_sample_2hop( static_cast<int*>(outputSRC1.data_ptr()),
                        static_cast<int*>(outputDST1.data_ptr()),
                        static_cast<int*>(outputSRC2.data_ptr()),
                        static_cast<int*>(outputDST2.data_ptr()),
                        (const int*) graphEdge.data_ptr(),
                        (const int*) boundList.data_ptr(),
                        (const int*) trainNode.data_ptr(),
                        sampleNUM1,sampleNUM2,nodeNUM,gpuDeviceIndex);
    //auto t_end = std::chrono::high_resolution_clock::now();
    //printf("sample2hop time in function`torch_launch_sample_2hop` : %lf ms\n",std::chrono::duration<double, std::milli>(t_end-t_beg).count());
    
}

void torch_launch_sample_3hop(torch::Tensor &outputSRC1,
                        torch::Tensor &outputDST1,
                        torch::Tensor &outputSRC2,
                        torch::Tensor &outputDST2,
                        torch::Tensor &outputSRC3,
                        torch::Tensor &outputDST3,
                       const torch::Tensor &graphEdge,
                       const torch::Tensor &boundList,
                       const torch::Tensor &trainNode,
                       int64_t sampleNUM1,
                       int64_t sampleNUM2,
                       int64_t sampleNUM3,
                       int64_t nodeNUM,
                       const int64_t gpuDeviceIndex) {
    launch_sample_3hop( static_cast<int*>(outputSRC1.data_ptr()),
                        static_cast<int*>(outputDST1.data_ptr()),
                        static_cast<int*>(outputSRC2.data_ptr()),
                        static_cast<int*>(outputDST2.data_ptr()),
                        static_cast<int*>(outputSRC3.data_ptr()),
                        static_cast<int*>(outputDST3.data_ptr()),
                        (const int*) graphEdge.data_ptr(),
                        (const int*) boundList.data_ptr(),
                        (const int*) trainNode.data_ptr(),
                        sampleNUM1,sampleNUM2,sampleNUM3,nodeNUM,gpuDeviceIndex);
}

void torch_launch_loading_halo(torch::Tensor &cacheData0,
                        torch::Tensor &cacheData1,
                        const torch::Tensor &edges,
                        const torch::Tensor &bound,
                        const int64_t cacheData0Len,
                        const int64_t cacheData1Len,
                        const int64_t edgesLen,
                        const int64_t boundLen,
                        const int64_t graphEdgeNUM,
                        const int64_t gpuDeviceIndex) {
      lanch_loading_halo(static_cast<int*>(cacheData0.data_ptr()),
                        static_cast<int*>(cacheData1.data_ptr()),
                        (const int*) edges.data_ptr(),
                        (const int*) bound.data_ptr(),
                        cacheData0Len,
                        cacheData1Len,
                        edgesLen,
                        boundLen,
                        graphEdgeNUM,
                        gpuDeviceIndex);
}

void torch_launch_loading_halo0(torch::Tensor &cacheData0,
                        torch::Tensor &cacheData1,
                        const torch::Tensor &edges,
                        const int64_t cacheData0Len,
                        const int64_t cacheData1Len,
                        const int64_t edgesLen,
                        const int64_t graphEdgeNUM,
                        const int64_t gpuDeviceIndex) {
      lanch_loading_halo0((int*) cacheData0.data_ptr(),
                        (int*) cacheData1.data_ptr(),
                        (const int*) edges.data_ptr(),
                        cacheData0Len,
                        cacheData1Len,
                        edgesLen,
                        graphEdgeNUM,
                        gpuDeviceIndex);
}

void readTensorFile(std::string output_file,torch::Tensor &output,int len) {
    FILE * fp = fopen64(output_file.c_str(),"r");
    assert(fp!=NULL);
    uint rd = 0;
    for(int i=0;i<len;i++) {
        int s;    
        rd += fread(&s, sizeof(int), 1, fp); 
        output[i] = s;
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_sample_full",
          &torch_launch_sample_full,
          "add2 kernel warpper");
    m.def("readTensorFile",
          &readTensorFile,
          "read tensor from file");
    m.def("torch_launch_sample_1hop",
          &torch_launch_sample_1hop,
          "sample for 1 hop");
    m.def("torch_launch_sample_2hop",
          &torch_launch_sample_2hop,
          "sample for 2 hop");
    m.def("torch_launch_sample_3hop",
          &torch_launch_sample_3hop,
          "sample for 3 hop");
    m.def("torch_launch_loading_halo",
            &torch_launch_loading_halo,
            "loading halo");
    m.def("torch_launch_loading_halo0",
            &torch_launch_loading_halo0,
            "loading halo");
    
    m.def("torch_launch_sample_1hop_new",
          &torch_launch_sample_1hop_new,
          "sample for 1 hop new");
    m.def("torch_launch_sample_2hop_new",
          &torch_launch_sample_2hop_new,
          "sample for 2 hop new");
    m.def("torch_launch_sample_3hop_new",
          &torch_launch_sample_3hop_new,
          "sample for 3 hop new");
    m.def("torch_launch_loading_halo_new",
            &torch_launch_loading_halo_new,
            "loading halo new");
}

// TORCH_LIBRARY(add2, m) {
//     m.def("torch_launch_sample_full", torch_launch_sample_full);
// }