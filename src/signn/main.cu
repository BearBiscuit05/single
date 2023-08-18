#include "common.cuh"
#include "cuda_hashtable.cuh"
#include "cuda_mapping.cuh"
#include "signn.h"

#define NUM 64

int main(){
  // OrderedHashTable<int64_t> table(80);
  // std::vector<int64_t> myVector = {2, 3, 4, 2, 7, 8};
  // myVector.resize(NUM,0);
  // std::vector<int64_t> myUnique(NUM,0);
  
  // int64_t inputNUM = NUM;
  // int64_t unique = 0;
  // table.FillWithDuplicates(myVector.data(),inputNUM,myUnique.data(),&unique);
  // std::cout << "uniqueNUM :"<< unique << std::endl;
  std::vector<int> edges(100,0);
  for (int i = 0 ; i < 100 ; i++) {
    edges[i] = i;
  }
  std::vector<int> bound= {0,2,10,20,20,21,30,40,40,50,50,60,60,70,70,80,80,90,90,100};
  std::vector<int> seed={0,1,2,3};
  int64_t seed_num=4;
  std::vector<int> fanouts={5};
  int64_t fanoutNUM=1;
  int sampledNUM = fanouts[0] * seed_num;
  std::vector<int> outSrcNodes(sampledNUM,0);
  std::vector<int> outDstNodes(sampledNUM,0);
  std::vector<int> outList(2,0);
  std::vector<int> outrawnodesid(sampledNUM,0);
  int64_t outnodesNUM = 0;

  int *dev_edges;int *dev_bound;int *dev_seed;
  int *dev_fanouts;int *dev_outSrcNodes;int *dev_outDstNodes;int *dev_outList;int *dev_outrawnodesid;
  printf("main in... \n");
  CUDA_CALL(cudaMalloc(&dev_edges, sizeof(int)*100));
  CUDA_CALL(cudaMalloc(&dev_bound, sizeof(int)*20));
  CUDA_CALL(cudaMalloc(&dev_seed, sizeof(int)*seed_num));
  CUDA_CALL(cudaMalloc(&dev_fanouts, sizeof(int)));
  CUDA_CALL(cudaMalloc(&dev_outSrcNodes, sizeof(int)*sampledNUM));
  CUDA_CALL(cudaMalloc(&dev_outDstNodes, sizeof(int)*sampledNUM));
  CUDA_CALL(cudaMalloc(&dev_outList, sizeof(int)*2));
  CUDA_CALL(cudaMalloc(&dev_outrawnodesid, sizeof(int)*sampledNUM));

	CUDA_CALL(cudaMemcpy(dev_edges, edges.data(), sizeof(int)*100, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_bound, bound.data(), sizeof(int)*20, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_seed, seed.data(), sizeof(int)*seed_num, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_fanouts, fanouts.data(), sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_outSrcNodes, outSrcNodes.data(), sizeof(int)*sampledNUM, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_outDstNodes, outDstNodes.data(), sizeof(int)*sampledNUM, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_outList, outList.data(), sizeof(int)*2, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_outrawnodesid, outrawnodesid.data(), sizeof(int)*sampledNUM, cudaMemcpyHostToDevice));
  printf("get in... \n");
  mutiLayersSample(
    dev_edges,dev_bound,
    dev_seed,seed_num,fanouts.data(),fanoutNUM,
    dev_outSrcNodes,dev_outDstNodes,dev_outList,
    dev_outrawnodesid,outnodesNUM);
  CUDA_CALL(cudaDeviceSynchronize());

  CUDA_CALL(cudaMemcpy(edges.data(),dev_edges, sizeof(int)*100, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(bound.data(),dev_bound,  sizeof(int)*20, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(seed.data(),dev_seed,  sizeof(int)*seed_num, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(fanouts.data(),dev_fanouts, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(outSrcNodes.data(),dev_outSrcNodes, sizeof(int)*sampledNUM, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(outDstNodes.data(),dev_outDstNodes,  sizeof(int)*sampledNUM, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(outList.data(), dev_outList, sizeof(int)*2, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(outrawnodesid.data(), dev_outrawnodesid,sizeof(int)*sampledNUM, cudaMemcpyDeviceToHost));
  std::cout << std::endl;

  for (int i = 0 ; i  < sampledNUM ; i++) {
    std::cout << outSrcNodes[i] << " ";
  }
  std::cout << std::endl;
  for (int i = 0 ; i  < sampledNUM ; i++) {
    std::cout << outDstNodes[i] << " ";
  }
  std::cout << std::endl;
  return 0;
}