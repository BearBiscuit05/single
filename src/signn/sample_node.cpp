#include "signn.h"


void torch_sample_hop(
    torch::Tensor &graphEdge,torch::Tensor &bound,
    torch::Tensor &seed,int seed_num,int fanout,
    torch::Tensor &out_src,torch::Tensor &out_dst,int gapNUM
    ) {
    sample_hop(
        (int*) graphEdge.data_ptr(),(int*) bound.data_ptr(),(int*) seed.data_ptr(),
        seed_num,fanout,(int*) out_src.data_ptr(),
        (int*) out_dst.data_ptr(),gapNUM);
}


void torch_graph_halo_merge(
    torch::Tensor &edge,torch::Tensor &bound,
    torch::Tensor &halos,torch::Tensor &halo_bound,int nodeNUM
) {
    graph_halo_merge(
        (int*) edge.data_ptr(),(int*) bound.data_ptr(),
        (int*) halos.data_ptr(),(int*) halo_bound.data_ptr(),nodeNUM
    );
}


void torch_graph_mapping(torch::Tensor &nodeList,torch::Tensor &mappingTable
                        ,int nodeNUM,int mappingNUM) {
    graph_mapping(
        (int*) nodeList.data_ptr(),(int*) mappingTable.data_ptr(),nodeNUM,mappingNUM
    );
}

void torch_mutiLayersSample(
    torch::Tensor & graphEdge,torch::Tensor &bound,
    torch::Tensor & seed,int64_t seed_num,torch::Tensor &fanouts,int64_t fanoutNUM,
    torch::Tensor & outSrcNodes,torch::Tensor & outDstNodes,torch::Tensor & outList,
    torch::Tensor & outrawnodesid,int64_t outnodesNUM
) {
    mutiLayersSample(
        (int*) graphEdge.data_ptr(),(int*) bound.data_ptr(),
        (int*) seed.data_ptr(), seed_num,(int*) fanouts.data_ptr(), fanoutNUM,
        (int*) outSrcNodes.data_ptr(),(int*) outDstNodes.data_ptr(),(int*) outList.data_ptr(),
        (int*) outrawnodesid.data_ptr(), outnodesNUM
    );

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_sample_hop",
          &torch_sample_hop,
          "sample neri");
    m.def("torch_graph_halo_merge",
          &torch_graph_halo_merge,
          "graph halo merge");
    m.def("torch_graph_mapping",
          &torch_graph_mapping,
          "mapping new graph");
    m.def("torch_mutiLayersSample",
          &torch_mutiLayersSample,
          "muti layer sample");
}

// TORCH_LIBRARY(sample_hop_new, m) {
//     m.def("torch_sample_hop", torch_sample_hop);
// }
