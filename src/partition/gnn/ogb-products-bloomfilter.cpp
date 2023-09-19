#include "bloomfilter.h"
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <set>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <sys/wait.h> 

extern int errno;

#define MAX_ITEMS 5000000    // 设置最大元素个数
//#define ADD_ITEMS 1000     // 添加测试元素
#define P_ERROR 0.000001     // 设置误差

/* ogb-products数据集全图边信息 */
const char* csvfile = "/home/wsy/single-gnn/data/ogb-products/edge.csv";
/* ogb-products数据集全图trainID信息 */
const char* trainIDfile = "/home/wsy/single-gnn/data/ogb-products/trainID.bin";
/* 输出的重要边信息目标文件 */
const char* selectedEdgesFile = "/home/wsy/single-gnn/data/ogb-products/important-edges.csv";

/* 设置迭代更新时候的权值 */
const int weight[2] = {50,20};
/* 设置迭代轮次，从训练节点出发几跳以内需要整 */
const int epochs = 2;

int main(int argc, char** argv)
{
	/* 1. 定义BaseBloomFilter */
	static BaseBloomFilter stBloomFilter = {0};

	/* 2. 初始化stBloomFilter，调用时传入hash种子，存储容量，以及允许的误判率 */
	InitBloomFilter(&stBloomFilter, 0, MAX_ITEMS, P_ERROR);

	/* 3. 初始时，将trainID先写进布隆过滤器 */
	printf("start to add trainIDs...\n");
	FILE* fp = fopen(trainIDfile, "rb");
	if(fp == NULL)
	{
		printf("open error\n");
		exit(-1);
	}
	char buf[4];
	while(!feof(fp))
	{
		if(1 == fread(buf,4,1,fp))
		{
			int id = *((int*)buf);
			if(0 != BloomFilter_Add(&stBloomFilter,(const void*)(&id),sizeof(int))){
				printf("add node %d failed\n",id);
			}
		}
	}
	fclose(fp);
	printf("complete adding trainIDs...\n");
	/* 4.1 重要的边权值更新直接写文件的文件句柄 */
	std::ofstream fout;
	fout.open(selectedEdgesFile,std::ios::out|std::ios::binary);
	if(!fout.is_open())
	{
		printf("open error\n");
	}

	/* 4.2 读边权csv的文件句柄 */
	std::ifstream fin;
	fin.open(csvfile,std::ios::in|std::ios::binary);
	if(!fin.is_open())
	{
		fout.close();
		printf("open error\n");
	}

	/* 5. 开始处理n跳节点及其边信息 */
	for(int i = 0;i < epochs;i++)
	{
		printf("epoch: %d\n",i+1);
		/* 维护的缓存列表 */
		std::set<int> nodes;

		int cnt = 0;
		int u,v;
		char ch;

		/* 每一轮从头开始读 */
		fin.seekg(0,std::ios::beg);
		/* 每次读一条边 */
		while(!fin.eof())
		{
			fin >> u >> ch >> v;
			//最后一行有空换行
			if(fin.good())
			{
				int ret = BloomFilter_CheckEdge(&stBloomFilter,u,v);
				if(ret == 1)
				{
					nodes.insert(u);
					fout << u << "," << v << "," << weight[i] << std::endl;
				}
				else if(ret == 2)
				{
					nodes.insert(v);
					fout << u << "," << v << "," << weight[i] << std::endl;
				}
				if(++cnt % 1000000 == 0)
				{
					printf("[epoch %d] deal with %d edges\n",i+1,cnt);
				}
			}
			else
			{
				fin.clear();
				break;
			}
		}

		printf("start adding cache nodes...\n");
		if(0 == BloomFilter_AddNodes(&stBloomFilter,nodes)){
			printf("add cache nodes success\n");
		}else{
			printf("add cache nodes failed\n");
		}
	}
	
	fin.close();
	fout.close();

	/* 6. 保存n跳节点生成的bloomfilter */
	char bffilename[128];
	sprintf(bffilename,"/home/wsy/single-gnn/data/ogb-products/bloomfilter%d.bin",epochs);
	if(0 != SaveBloomFilterToFile(&stBloomFilter,bffilename))
	{
		printf("failed to save bloomfilter.\n");
	}
	/* 7. 释放布隆过滤器 */
	FreeBloomFilter(&stBloomFilter);
	printf("End....\n");
	return 0;
}