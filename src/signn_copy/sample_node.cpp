#include "signn.h"


void torch_sample_hop(
    torch::Tensor &graphEdge,torch::Tensor &bound,
    torch::Tensor &seed,int seed_num,int fanout,
    torch::Tensor &out_src,torch::Tensor &out_dst,torch::Tensor &num_out
    ) {
    sample_hop(
        (int*) graphEdge.data_ptr(),(int*) bound.data_ptr(),(int*) seed.data_ptr(),
        seed_num,fanout,(int*) out_src.data_ptr(),
        (int*) out_dst.data_ptr(),(size_t*) num_out.data_ptr());
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


void torch_graph_mapping(
    torch::Tensor &nodeList,torch::Tensor &nodeSRC,
    torch::Tensor &nodeDST,torch::Tensor &newNodeSRC,
    torch::Tensor &newNodeDST,torch::Tensor &uniqueList,
    int edgeNUM,int uniqueNUM) {
    graph_mapping(
        (int*) nodeList.data_ptr(),(int*) nodeSRC.data_ptr(),
        (int*) nodeDST.data_ptr(),(int*) newNodeSRC.data_ptr(),
        (int*) newNodeDST.data_ptr(),(int*) uniqueList.data_ptr(),
        edgeNUM,uniqueNUM
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
}

// TORCH_LIBRARY(sample_hop_new, m) {
//     m.def("torch_sample_hop", torch_sample_hop);
// }
