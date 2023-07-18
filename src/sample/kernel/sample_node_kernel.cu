#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <chrono>
#include <numeric>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>

__global__ void sample_full_kernel(
                            int* outputSRC,
                            int* outputDST,
                            const int* graphEdge,
                            const int* boundList,
                            const int* trainNode,
                            int nodeNUM) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx ; i < nodeNUM ; i += blockDim.x) {
        int writeIdx = i * 25;
        int id = trainNode[i];
        int idxStart = boundList[id];
        int idxEnd = boundList[id+1];
        for (int l = 0 ; l < (idxEnd - idxStart) ; l++) {
            outputSRC[writeIdx] = graphEdge[idxStart + l];
            outputDST[writeIdx++] = id;
        }

    }    
}

__global__ void sample1Hop(
                        int* outputSRC1,
                        int* outputDST1, 
                        const int* graphEdge,
                        const int* boundList,
                        const int* trainNode,
                        int sampleNUM1,
                        int nodeNUM,
                        unsigned long long seed
                            ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandStateXORWOW_t state;
    curand_init(seed+idx,0,0,&state);
    unsigned int random_value = 0;
    int blockSize = sampleNUM1;
    for(int i = idx ; i < nodeNUM ; i += blockDim.x) {
        int writeIdx = i * blockSize;
        int id = trainNode[i];
        int idxStart = boundList[id];
        int idxEnd = boundList[id+1];
        int neirNUM = idxEnd - idxStart;
        for (int l = 0 ; l < neirNUM ; l++) { 
            random_value = curand(&state) % neirNUM;
            outputSRC1[writeIdx] = graphEdge[idxStart + random_value];
            outputDST1[writeIdx++] = id;
        }
        for (int l = neirNUM; l < sampleNUM1 ; l++) {
            outputSRC1[writeIdx] = 0;
            outputDST1[writeIdx++] = id;
        }
    }
}

__global__ void sample2Hop(
                        int* outputSRC1,
                        int* outputDST1,
                        int* outputSRC2,
                        int* outputDST2,
                        const int* graphEdge,
                        const int* boundList,
                        const int* trainNode,
                        int sampleNUM1,
                        int sampleNUM2,
                        int nodeNUM,
                        unsigned long long seed
                            ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandStateXORWOW_t state;
    curand_init(seed+idx,0,0,&state);
    unsigned int random_value = 0;
    for(int i = idx ; i < nodeNUM ; i += blockDim.x) {
        int writeIdx = i * sampleNUM1;
        int id = trainNode[i];
        int idxStart = boundList[id];
        int idxEnd = boundList[id+1];
        int neirNUM = idxEnd - idxStart;
        for (int l = 0 ; l < neirNUM && l < sampleNUM1 ; l++) {
            random_value = curand(&state) % neirNUM;
            outputSRC1[writeIdx] = graphEdge[idxStart + random_value];
            outputDST1[writeIdx++] = id;
        }
        for (int l = neirNUM; l < sampleNUM1 ; l++) {
            outputSRC1[writeIdx] = -1;
            outputDST1[writeIdx++] = id;
        }

        // hop-2
        for (int l1 = 0 ; l1 < sampleNUM1 ; l1++) {
            // 二层采样id
            int l2_id = outputSRC1[i * sampleNUM1 + l1];
            if (l2_id > 0) {
                int l2_writeIdx = i*sampleNUM1*sampleNUM2 + l1*sampleNUM2;
                int l2_idStart = boundList[l2_id];
                int l2_idEnd = boundList[l2_id+1];
                int l2_neirNUM = l2_idEnd - l2_idStart;
                for (int l = 0 ; l < l2_neirNUM && l < sampleNUM2 ; l++) {
                    random_value = curand(&state) % l2_neirNUM;
                    outputSRC2[l2_writeIdx] = graphEdge[l2_idStart + random_value];
                    outputDST2[l2_writeIdx++] = l2_id;
                }
                for (int l = l2_neirNUM; l < sampleNUM2 ; l++) {
                    outputSRC2[l2_writeIdx] = -1;
                    outputDST2[l2_writeIdx++] = l2_id;
                }
            } else {
                int l2_writeIdx = i*sampleNUM1*sampleNUM2 + l1*sampleNUM2;
                for (int l = 0 ; l < sampleNUM2 ; l++) {
                    outputSRC2[l2_writeIdx] = -1;
                    outputDST2[l2_writeIdx++] = l2_id;
                }
            }
        }
    }
}

__global__ void sample3Hop(
                        int* outputSRC1,int* outputDST1,
                        int* outputSRC2,int* outputDST2,
                        int* outputSRC3,int* outputDST3,
                        const int* graphEdge,
                        const int* boundList,
                        const int* trainNode,
                        int sampleNUM1,int sampleNUM2,int sampleNUM3,
                        int nodeNUM,
                        unsigned long long seed
                            ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandStateXORWOW_t state;
    curand_init(seed+idx,0,0,&state);
    unsigned int random_value = 0;
    for(int i = idx ; i < nodeNUM ; i += blockDim.x) {
        int writeIdx = i * sampleNUM1;
        int id = trainNode[i];
        int idxStart = boundList[id];
        int idxEnd = boundList[id+1];
        int neirNUM = idxEnd - idxStart;
        for (int l = 0 ; l < neirNUM && l < sampleNUM1 ; l++) {
            random_value = curand(&state) % neirNUM;
            outputSRC1[writeIdx] = graphEdge[idxStart + random_value];
            outputDST1[writeIdx++] = id;
        }
        for (int l = neirNUM; l < sampleNUM1 ; l++) {
            outputSRC1[writeIdx] = -1;
            outputDST1[writeIdx++] = id;
        }

        // hop-2
        for (int l1 = 0 ; l1 < sampleNUM1 ; l1++) {
            // 二层采样id
            int l2_id = outputSRC1[i * sampleNUM1 + l1];
            if (l2_id > 0) {
                int l2_writeIdx = i*sampleNUM1*sampleNUM2 + l1*sampleNUM2;
                int l2_idStart = boundList[l2_id];
                int l2_idEnd = boundList[l2_id+1];
                int l2_neirNUM = l2_idEnd - l2_idStart;
                for (int l = 0 ; l < l2_neirNUM && l < sampleNUM2 ; l++) {
                    random_value = curand(&state) % l2_neirNUM; 
                    outputSRC2[l2_writeIdx] = graphEdge[l2_idStart + random_value];
                    outputDST2[l2_writeIdx++] = l2_id;
                }
                for (int l = l2_neirNUM; l < sampleNUM2 ; l++) {
                    outputSRC2[l2_writeIdx] = -1;
                    outputDST2[l2_writeIdx++] = l2_id;
                }
            } else {
                int l2_writeIdx = i*sampleNUM1*sampleNUM2 + l1*sampleNUM2;
                for (int l = 0 ; l < sampleNUM2 ; l++) {
                    outputSRC2[l2_writeIdx] = -1;
                    outputDST2[l2_writeIdx++] = l2_id;
                }
            }
            
        }

        for (int l2 = 0 ; l2 < sampleNUM2 ; l2++) {
            int l3_id = outputSRC2[i * sampleNUM2 + l2];
            if (l3_id > 0) {
                int l3_writeIdx = i*sampleNUM2*sampleNUM3 + l2*sampleNUM3;
                int l3_idStart = boundList[l3_id];
                int l3_idEnd = boundList[l3_id+1];
                int l3_neirNUM = l3_idEnd - l3_idStart;
                for (int l = 0 ; l < l3_neirNUM && l < sampleNUM3 ; l++) {
                    random_value = curand(&state) % l3_neirNUM; 
                    outputSRC3[l3_writeIdx] = graphEdge[l3_idStart + random_value];
                    outputDST3[l3_writeIdx++] = l3_id;
                }
                for (int l = l3_neirNUM; l < sampleNUM3 ; l++) {
                    outputSRC3[l3_writeIdx] = -1;
                    outputDST3[l3_writeIdx++] = l3_id;
                }
            } else {
                int l3_writeIdx = i*sampleNUM2*sampleNUM3 + l2*sampleNUM3;
                for (int l = 0 ; l < sampleNUM2 ; l++) {
                    outputSRC3[l3_writeIdx] = -1;
                    outputDST3[l3_writeIdx++] = l3_id;
                }
            }
        }
    }
}

void launch_sample_full(int* outputSRC1,
                 int* outputDST1,
                 const int* graphEdge,
                 const int* boundList,
                 const int* trainNode,
                 int n,
                 const int gpuDeviceIndex
                 ) {
    dim3 grid((n + 1023) / 1024);
    dim3 block(1024);
    
    /* 指定使用的GPU序号 [0,torch.cuda.device_count()) */
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No GPU devices found.\n");
        return;
    }
    else if(gpuDeviceIndex >= deviceCount || gpuDeviceIndex < 0){
        printf("Wrong GPU Device Index:%d , Select Default Device Index:0 cuda:0.\n",gpuDeviceIndex);
        cudaSetDevice(0);
    }
    else{
        printf("Select GPU Device Index:%d , Please Prepare Pytorch Data tensor.to(device='cuda:%d')\n",gpuDeviceIndex,gpuDeviceIndex);
        cudaSetDevice(gpuDeviceIndex);
    }

    sample_full_kernel<<<grid, block>>>(outputSRC1, outputDST1, graphEdge, boundList, trainNode, n);
}

void launch_sample_1hop(int* outputSRC1,
                        int* outputDST1, 
                        const int* graphEdge,
                        const int* boundList,
                        const int* trainNode,
                        int sampleNUM1,
                        int nodeNUM,
                        const int gpuDeviceIndex
                        ) {
    dim3 grid((nodeNUM + 1023) / 1024);
    dim3 block(1024);
    unsigned long long seed = std::chrono::system_clock::now().time_since_epoch().count();
    
    /* 指定使用的GPU序号 [0,torch.cuda.device_count()) */
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No GPU devices found.\n");
        return;
    }
    else if(gpuDeviceIndex >= deviceCount || gpuDeviceIndex < 0){
        printf("Wrong GPU Device Index:%d , Select Default Device Index:0 cuda:0.\n",gpuDeviceIndex);
        cudaSetDevice(0);
    }
    else{
        printf("Select GPU Device Index:%d , Please Prepare Pytorch Data tensor.to(device='cuda:%d')\n",gpuDeviceIndex,gpuDeviceIndex);
        cudaSetDevice(gpuDeviceIndex);
    }

    sample1Hop<<<grid, block>>>(
        outputSRC1,outputDST1,graphEdge,
        boundList,trainNode,sampleNUM1,
        nodeNUM,seed);
}

void launch_sample_2hop(int* outputSRC1,
                        int* outputDST1,
                        int* outputSRC2,
                        int* outputDST2,
                        const int* graphEdge,
                        const int* boundList,
                        const int* trainNode,
                        int sampleNUM1,
                        int sampleNUM2,
                        int nodeNUM,
                        const int gpuDeviceIndex
                        ) {
    dim3 grid((nodeNUM + 1023) / 1024);
    dim3 block(1024);
    unsigned long long seed = std::chrono::system_clock::now().time_since_epoch().count();

    /* 指定使用的GPU序号 [0,torch.cuda.device_count()) */
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No GPU devices found.\n");
        return;
    }
    else if(gpuDeviceIndex >= deviceCount || gpuDeviceIndex < 0){
        printf("Wrong GPU Device Index:%d , Select Default Device Index:0 cuda:0.\n",gpuDeviceIndex);
        cudaSetDevice(0);
    }
    else{
        printf("Select GPU Device Index:%d , Please Prepare Pytorch Data tensor.to(device='cuda:%d')\n",gpuDeviceIndex,gpuDeviceIndex);
        cudaSetDevice(gpuDeviceIndex);
    }

    sample2Hop<<<grid, block>>>(
        outputSRC1,outputDST1,outputSRC2,
        outputDST2,graphEdge,boundList,
        trainNode,sampleNUM1,sampleNUM2,nodeNUM,seed);
}

void launch_sample_3hop(int* outputSRC1,int* outputDST1,
                        int* outputSRC2,int* outputDST2,
                        int* outputSRC3,int* outputDST3,
                        const int* graphEdge,
                        const int* boundList,
                        const int* trainNode,
                        int sampleNUM1,int sampleNUM2,int sampleNUM3,
                        int nodeNUM,
                        const int gpuDeviceIndex
                        ) {
    dim3 grid((nodeNUM + 1023) / 1024);
    dim3 block(1024);
    unsigned long long seed = std::chrono::system_clock::now().time_since_epoch().count();

    /* 指定使用的GPU序号 [0,torch.cuda.device_count()) */
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No GPU devices found.\n");
        return;
    }
    else if(gpuDeviceIndex >= deviceCount || gpuDeviceIndex < 0){
        printf("Wrong GPU Device Index:%d , Select Default Device Index:0 cuda:0.\n",gpuDeviceIndex);
        cudaSetDevice(0);
    }
    else{
        printf("Select GPU Device Index:%d , Please Prepare Pytorch Data tensor.to(device='cuda:%d')\n",gpuDeviceIndex,gpuDeviceIndex);
        cudaSetDevice(gpuDeviceIndex);
    }

    sample3Hop<<<grid, block>>>(
        outputSRC1,outputDST1,outputSRC2,
        outputDST2,outputSRC3,outputDST3,
        graphEdge,boundList,trainNode,
        sampleNUM1,sampleNUM2,sampleNUM3,nodeNUM,seed);
}