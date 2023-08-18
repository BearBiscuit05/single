#include <stdio.h>    
#include <stdlib.h>   
#include <cuda_runtime.h>  
 
#define SIZE 8

__device__ void device_kernel(unsigned int *histo,int i) 
{
	atomicAdd(histo, i);
}


__global__ void histo_kernel(int size, unsigned int *histo)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < size)
	{
		device_kernel(histo, i);
	}
}

__global__ void histo(int size, unsigned int *histo)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < size)
	{
		device_kernel(histo, i);
	}
}

int main(void)
{
	int threadSum = 0;
 
	//分配内存并拷贝初始数据
	unsigned int *dev_histo;
 
	cudaMalloc((void**)&dev_histo, sizeof(int));
	cudaMemcpy(dev_histo, &threadSum, sizeof(int), cudaMemcpyHostToDevice);
 
	// kernel launch - 2x the number of mps gave best timing  
	cudaDeviceProp  prop;
	cudaGetDeviceProperties(&prop, 0);
 
	int blocks = 2;
	//确保线程数足够
	histo_kernel << <blocks * 2, (SIZE + 2 * blocks - 1) / blocks / 2 >> > (SIZE, dev_histo);
	cudaMemcpy(&threadSum, dev_histo, sizeof(int), cudaMemcpyDeviceToHost);
	printf("Threads SUM：%d\n", threadSum);
	histo <<<blocks * 2, (SIZE + 2 * blocks - 1) / blocks / 2 >>> (SIZE, dev_histo);
	printf("Threads SUM：%d\n", threadSum);
	//数据拷贝回CPU内存
	cudaMemcpy(&threadSum, dev_histo, sizeof(int), cudaMemcpyDeviceToHost);
	printf("Threads SUM：%d\n", threadSum);
	cudaFree(dev_histo);
	return 0;
}