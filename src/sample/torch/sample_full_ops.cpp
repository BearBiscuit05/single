#include <torch/extension.h>
#include "sample_full.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <map>
#include <sstream>
#include <cassert>

void torch_launch_sample_full(torch::Tensor &outputSRC1,
                        torch::Tensor &outputDST1,
                       const torch::Tensor &graphEdge,
                       const torch::Tensor &boundList,
                       const torch::Tensor &trainNode,
                       int64_t nodeNUM) {
    launch_sample_full((int*) outputSRC1.data_ptr(),
                 (int*) outputDST1.data_ptr(),
                 (const int*) graphEdge.data_ptr(),
                 (const int*) boundList.data_ptr(),
                 (const int*) trainNode.data_ptr(),
                 nodeNUM) ;
}

void torch_launch_sample_1hop(torch::Tensor &outputSRC1,
                        torch::Tensor &outputDST1,
                       const torch::Tensor &graphEdge,
                       const torch::Tensor &boundList,
                       const torch::Tensor &trainNode,
                       int64_t sampleNUM1,
                       int64_t nodeNUM) {
    launch_sample_1hop((int *)outputSRC1.data_ptr(),
                    (int *)outputDST1.data_ptr(), 
                    (const int *)graphEdge.data_ptr(),
                    (const int *)boundList.data_ptr(),
                    (const int *)trainNode.data_ptr(),
                    sampleNUM1,
                    nodeNUM);
    
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
                       int64_t nodeNUM) {
    launch_sample_2hop((int*) outputSRC1.data_ptr(),
                        (int*) outputDST1.data_ptr(),
                        (int* )outputSRC2.data_ptr(),
                        (int*) outputDST2.data_ptr(),
                        (const int*) graphEdge.data_ptr(),
                        (const int*) boundList.data_ptr(),
                        (const int*) trainNode.data_ptr(),
                        sampleNUM1,sampleNUM2,nodeNUM);
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
                       int64_t nodeNUM) {
    launch_sample_3hop((int*) outputSRC1.data_ptr(),
                        (int*) outputDST1.data_ptr(),
                        (int* )outputSRC2.data_ptr(),
                        (int*) outputDST2.data_ptr(),
                        (int*) outputSRC3.data_ptr(),
                        (int*) outputDST3.data_ptr(),
                        (const int*) graphEdge.data_ptr(),
                        (const int*) boundList.data_ptr(),
                        (const int*) trainNode.data_ptr(),
                        sampleNUM1,sampleNUM2,sampleNUM3,nodeNUM);
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
}

// TORCH_LIBRARY(add2, m) {
//     m.def("torch_launch_sample_full", torch_launch_sample_full);
// }