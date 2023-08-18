#include "common.cuh"
#include "cuda_hashtable.cuh"
#include "cuda_mapping.cuh"
#include "signn.h"

#define NUM 64


void graph_mapping(
    int* nodeList,int* nodeSRC,
    int* nodeDST,int* newNodeSRC,
    int* newNodeDST,int* uniqueList,
    int edgeNUM,int uniqueNUM
) {
  OrderedHashTable<int> table(edgeNUM*2);
  int64_t* dev_num_unique;
  int64_t num_unique=0;
  size_t nodeNUM = edgeNUM * 2;
  CUDA_CALL(cudaMemcpy(dev_num_unique, &num_unique,sizeof(int64_t), cudaMemcpyHostToDevice));
  table.FillWithDuplicates(nodeList,nodeNUM,uniqueList,dev_num_unique);
  CUDA_CALL(cudaMemcpy(&num_unique,dev_num_unique, sizeof(int64_t), cudaMemcpyDeviceToHost));
  uniqueNUM = (int) num_unique;
  

  GPUMapEdges<int>(nodeSRC, newNodeSRC,
                  nodeDST, newNodeDST,
                  edgeNUM, table.DeviceHandle()
                );
  
  

}





int main(){
  std::vector<int> edges(100,0);
  for (int i = 0 ; i < 100 ; i++) {
    edges[i] = i;
  }
  std::vector<int> bound= {0,2,10,20,20,21,30,40,40,50,50,60,60,70,70,80,80,90,90,100};
  std::vector<int> seed={0,1,2,3};
  int64_t seed_num=4;
  std::vector<int> fanouts={5};
  int sampledNUM = fanouts[0] * seed_num;
  std::vector<int> outSrcNodes(sampledNUM,0);
  std::vector<int> outDstNodes(sampledNUM,0);
  std::vector<int> outList(2,0);
  std::vector<int> outrawnodesid(sampledNUM,0);
  std::vector<size_t> outNUM(1,0);

  int *dev_edges;int *dev_bound;int *dev_seed;size_t* dev_outNUM;
  int *dev_fanouts;int *dev_outSrcNodes;int *dev_outDstNodes;int *dev_outList;int *dev_outrawnodesid;

  CUDA_CALL(cudaMalloc(&dev_edges, sizeof(int)*100));
  CUDA_CALL(cudaMalloc(&dev_bound, sizeof(int)*20));
  CUDA_CALL(cudaMalloc(&dev_seed, sizeof(int)*seed_num));
  CUDA_CALL(cudaMalloc(&dev_fanouts, sizeof(int)));
  CUDA_CALL(cudaMalloc(&dev_outSrcNodes, sizeof(int)*sampledNUM));
  CUDA_CALL(cudaMalloc(&dev_outDstNodes, sizeof(int)*sampledNUM));
  CUDA_CALL(cudaMalloc(&dev_outList, sizeof(int)*2));
  CUDA_CALL(cudaMalloc(&dev_outrawnodesid, sizeof(int)*sampledNUM));
  CUDA_CALL(cudaMalloc(&dev_outNUM, sizeof(size_t)));

	CUDA_CALL(cudaMemcpy(dev_edges, edges.data(), sizeof(int)*100, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_bound, bound.data(), sizeof(int)*20, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_seed, seed.data(), sizeof(int)*seed_num, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_fanouts, fanouts.data(), sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_outSrcNodes, outSrcNodes.data(), sizeof(int)*sampledNUM, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_outDstNodes, outDstNodes.data(), sizeof(int)*sampledNUM, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_outList, outList.data(), sizeof(int)*2, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_outrawnodesid, outrawnodesid.data(), sizeof(int)*sampledNUM, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_outNUM, outNUM.data(), sizeof(size_t), cudaMemcpyHostToDevice));

  int fanout = fanouts[0];
  sample_hop(
    dev_edges,dev_bound,dev_seed,
    seed_num,fanout,dev_outSrcNodes,
    dev_outDstNodes,dev_outNUM);

  CUDA_CALL(cudaDeviceSynchronize());

  CUDA_CALL(cudaMemcpy(edges.data(),dev_edges, sizeof(int)*100, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(bound.data(),dev_bound,  sizeof(int)*20, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(seed.data(),dev_seed,  sizeof(int)*seed_num, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(fanouts.data(),dev_fanouts, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(outSrcNodes.data(),dev_outSrcNodes, sizeof(int)*sampledNUM, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(outDstNodes.data(),dev_outDstNodes,  sizeof(int)*sampledNUM, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(outList.data(), dev_outList, sizeof(int)*2, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(outrawnodesid.data(), dev_outrawnodesid,sizeof(int)*sampledNUM, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(outNUM.data(), dev_outNUM,sizeof(size_t), cudaMemcpyDeviceToHost));
  std::cout << std::endl;

  for (int i = 0 ; i  < sampledNUM ; i++) {
    std::cout << outSrcNodes[i] << " ";
  }
  std::cout << std::endl;
  for (int i = 0 ; i  < sampledNUM ; i++) {
    std::cout << outDstNodes[i] << " ";
  }
  std::cout << std::endl;
  printf("outNUM: %zu \n",outNUM[0]);
  std::cout << std::endl;


  
  // ===================================================>
  int* dev_mergedVector;
  std::vector<int> mergedVector(outDstNodes.begin(), outDstNodes.begin()+outNUM[0]);
  mergedVector.insert(mergedVector.end(), outSrcNodes.begin(), outSrcNodes.begin()+outNUM[0]);
  outNUM[0] = outNUM[0] * 2;
  cudaMalloc(&dev_mergedVector, sizeof(int)*outNUM[0]);
	cudaMemcpy(dev_mergedVector, mergedVector.data(), sizeof(int)*outNUM[0], cudaMemcpyHostToDevice);

  OrderedHashTable<int> table(outNUM[0]);
  std::vector<int> myUnique(outNUM[0],0);
  int *dev_myUnique;
  CUDA_CALL(cudaMalloc(&dev_myUnique, sizeof(int)*outNUM[0]));
  CUDA_CALL(cudaMemcpy(dev_myUnique, myUnique.data(), sizeof(int)*outNUM[0], cudaMemcpyHostToDevice));

  int64_t* dev_num_unique;
  int64_t num_unique=0;
  cudaMalloc(&dev_num_unique, sizeof(int64_t));
	cudaMemcpy(dev_num_unique, &num_unique, sizeof(int64_t), cudaMemcpyHostToDevice);
  table.FillWithDuplicates(dev_mergedVector,outNUM[0],dev_myUnique,dev_num_unique);

  CUDA_CALL(cudaMemcpy(myUnique.data(),dev_myUnique, sizeof(int)*outNUM[0], cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(&num_unique,dev_num_unique, sizeof(int64_t), cudaMemcpyDeviceToHost));
  std::cout << "uniqueNUM :"<< num_unique << std::endl;
  for (int i = 0 ; i  < num_unique ; i++) {
    std::cout <<"unique :" << myUnique[i] << " ";
  }
  std::cout << std::endl;

  
  std::vector<int> new_global_src(outNUM[0],0);
  std::vector<int> new_global_dst(outNUM[0],0);
  int * dev_new_global_src;int * dev_new_global_dst;
  CUDA_CALL(cudaMalloc(&dev_new_global_src, sizeof(int)*outNUM[0]));
  CUDA_CALL(cudaMemcpy(dev_new_global_src, new_global_src.data(), sizeof(int)*outNUM[0], cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&dev_new_global_dst, sizeof(int)*outNUM[0]));
  CUDA_CALL(cudaMemcpy(dev_new_global_dst, new_global_dst.data(), sizeof(int)*outNUM[0], cudaMemcpyHostToDevice));

  GPUMapEdges<int>(dev_outSrcNodes, dev_new_global_src,
                  dev_outDstNodes, dev_new_global_dst,
                  outNUM[0], table.DeviceHandle()
                );
  CUDA_CALL(cudaMemcpy(new_global_src.data(),dev_new_global_src, sizeof(int)*outNUM[0], cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(new_global_dst.data(),dev_new_global_dst, sizeof(int)*outNUM[0], cudaMemcpyDeviceToHost));
  
  std::cout << "-----------------------------" << std::endl;
  for (int i = 0 ; i  < outNUM[0] / 2 ; i++) {
    std::cout << "new edge:" << new_global_src[i] << "-->" << new_global_dst[i] << "\n";
  }
  return 0;
}