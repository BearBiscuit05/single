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

void readSrcList(const char* datasetPath,const int partid)
{
	char filename[128];
	sprintf(filename,"%s/part%d/srcList.bin",datasetPath,partid);

	FILE* fp1 = fopen(filename, "rb");
	if(fp1 == NULL)
	{
		printf("open error\n");
		exit(-1);
	}

	sprintf(filename,"%s/part%d/range.bin",datasetPath,partid);
	FILE* fp2 = fopen(filename, "rb");
	if(fp2 == NULL)
	{
		fclose(fp1);
		printf("open error\n");
		exit(-1);
	}

	char buf[8];
	int boundl,boundr;
	int pos = 0;
	while(!feof(fp2))
	{
		if(2 == fread(buf,4,2,fp2))
		{
			boundl = *((int*)buf);
			boundr = *((int*)(buf+4));
			if(boundl == boundr)
			{
				printf("\n");
				continue;
			}
			int len = boundr-boundl;
			int* srcids = new(std::nothrow) int[len];
			if(srcids == NULL)
			{
				printf("new error!\n");
			}
			fseek(fp1,sizeof(int)*boundl,SEEK_SET);
			if(len == fread((char*)srcids,sizeof(int),len,fp1))
			{
				printf("%d: ",pos++);
				for(int k = 0;k < len;k++)
					printf("%d ",srcids[k]);
				printf("\n");
			}
			delete[] srcids;
		}
		else
		{
			break;//printf("read error!\n");
		}
		// printf("%d: [%d:%d]\n",pos++,boundl,boundr);
	}
	fclose(fp1);
	fclose(fp2);
}

void readHalo(const char* datasetPath,const int traingGID,const int nextGID)
{
	char filename[128];
	sprintf(filename,"%s/part%d/halo%d.bin",datasetPath,traingGID,nextGID);

	FILE* fp1 = fopen(filename, "rb");
	if(fp1 == NULL)
	{
		printf("open error\n");
		exit(-1);
	}

	sprintf(filename,"%s/part%d/halo%d_bound.bin",datasetPath,traingGID,nextGID);
	FILE* fp2 = fopen(filename, "rb");
	if(fp2 == NULL)
	{
		fclose(fp1);
		printf("open error\n");
		exit(-1);
	}

	char buf[4];
	int boundl = 0;
	int boundr = 0;
	int pos = 0;
	fread(buf,4,1,fp2);
	while(!feof(fp2))
	{
		boundl = boundr;
		if(1 == fread(buf,4,1,fp2))
		{
			boundr = *((int*)buf);
			if(boundl == boundr)
			{
				printf("\n");
				pos++;
				continue;
			}
			
			int len = boundr-boundl;
			int* srcids = new(std::nothrow) int[len];
			if(srcids == NULL)
			{
				printf("new error!\n");
			}
			fseek(fp1,sizeof(int)*boundl,SEEK_SET);
			if(len == fread((char*)srcids,sizeof(int),len,fp1))
			{
				printf("%d: ",pos);
				std::vector<int> halo;
				for(int k = 0;k < len;k++)
					if(k % 2 == 0)
						halo.push_back(srcids[k]);
					else if(srcids[k] != pos)
						printf("halo bound error!\n");
				for(int srcid:halo)
				{
					printf("%d ",srcid);
				}
				printf("\n");
				pos++;
			}
			delete[] srcids;
		}
		else
		{
			break;
		}
	}
	fclose(fp1);
	fclose(fp2);
}

// void readTrainID(const char* datasetPath,const int traingGID,std::vector<int64_t>& trainIDs)
// {
// 	char filename[128];
// 	sprintf(filename,"%s/trainID%d.txt",datasetPath,traingGID);
// 	std::ifstream fin;
// 	fin.open(filename,std::ios::in|std::ios::binary);
// 	int64_t id;
// 	while(!fin.eof())
// 	{
// 		fin >> id;
// 		trainIDs.push_back(id);
// 	}
// }

void readTrainID(const char* binfile,std::vector<int>& trainIds)
{
	FILE* fp = fopen(binfile, "rb");
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
			trainIds.push_back(id);
		}
	}
	fclose(fp);
}

void readEdgesCSV(const char* csvfile)
{
	std::ifstream fin;
	fin.open(csvfile,std::ios::in|std::ios::binary);
	if(!fin.is_open())
	{
		printf("open error\n");
	}
	int u,v;
	char ch;
	int cnt = 0;
	while(!fin.eof())
	{
		fin >> u >> ch >> v;
		//最后一行有空换行
		if(fin.good())
			cnt++;
	}
	fin.close();
	printf("%d edges in %s\n",cnt,csvfile);
}

int main()
{
	/* 测试读取分区的srcList.bin和range.bin */
	//readSrcList("/home/bear/workspace/singleGNN/data/products_4",0);

	/* 测试读取分区的halo.bin和halo_bound.bin */
	//readHalo("/home/bear/workspace/singleGNN/data/products_4",0,1);

	/* 测试读取分区trainID文件 */
	// std::vector<int64_t> trainIDs;
	// readTrainID(".",0,trainIDs);
	// for(int64_t id:trainIDs)
	// {
	// 	printf("%ld\n",id);
	// }

	/* 测试读取trainID的二进制文件 */
	// std::vector<int> trainIDs;
	// readTrainID("/home/wsy/single-gnn/data/ogb-products-trainID.bin",trainIDs);
	// for(int id:trainIDs)
	// {
	// 	printf("%d\n",id);
	// }

	/* 测试读取csv文件，按照srcid,dstid的格式 
		注意：python加载的边数是csv文件的2倍，因为dgl认为是有2条有向边
	*/
	readEdgesCSV("/home/wsy/single-gnn/data/ogb-products-edge.csv");
	return 0;
}