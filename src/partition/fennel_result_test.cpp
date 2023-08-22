/* 检查，fennel图分区以后，权值提高的重要边被割的比例 */
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
using namespace std;

/* 全图节点数 */
const int nodeNum = 2449029;
/* 重要边信息目标文件 */
const char* selectedEdgesFile = "/home/wsy/single-gnn/data/ogb-products/important-edges.csv";
/* 输出的fennel图分区结果 */
const char* partitionFile = "/home/wsy/single-gnn/data/ogb-products/result";
/* ogb-products数据集全图边信息 */
const char* allEdgesfile = "/home/wsy/single-gnn/data/ogb-products/edge.csv";
/* 存放每一个node对应在哪一个分区 */
char part[nodeNum];

/* 0,1,2分别表示其他节点，1跳节点，2跳节点 */
int bound[3] = {0,0,0};
int cut[3] = {0,0,0};

void readRes()
{
	ifstream fin;
	fin.open(partitionFile,ios::in|ios::binary);
	if(!fin.is_open())
	{
		cerr <<  "open error!" << endl;
		exit(-1);
	}
	while(!fin.eof())
	{
		int partid,size,id;
		fin >> partid >> size;
		if(!fin.good())
			break;
		cout << "part " << partid << " has " << size << " nodes" << endl;
		for(int i = 0;i < size;i++)
		{
			fin >> id;
			part[id] = '0' + partid;
		}
	}
}

/* 读取有边权的重要边 */
void readImportantEdges()
{
	ifstream fin;
	fin.open(selectedEdgesFile,ios::in|ios::binary);
	if(!fin.is_open())
	{
		cerr <<  "open error!" << endl;
		exit(-1);
	}
	int srcid,dstid,weight;
	char ch1,ch2;
	
	while(!fin.eof())
	{
		fin >> srcid >> ch1 >> dstid >> ch2 >> weight;
		if(fin.good())
		{
			int b = -1;
			if(weight == 50)
				b = 1;
			else if(weight == 20)
				b = 2;
			bound[b]++;
			if(part[srcid] != part[dstid])
				cut[b]++;
		}
		else{
			fin.clear();
			break;
		}
	}
	fin.close();
	double cuttingRate[2] = {
		100.00 * double(cut[1]) / double(bound[1]),
		100.00 * double(cut[2]) / double(bound[2]),
	};
	cout << "cutting rate of bound 1 edges:" << cuttingRate[0] << "%" << endl;
	cout << "cutting rate of bound 2 edges:" << cuttingRate[1] << "%" << endl;
}

/* 读取没有边权的原文件，先读重要边 */
void readCommonEdges()
{
	ifstream fin;
	fin.open(allEdgesfile,ios::in|ios::binary);
	if(!fin.is_open())
	{
		cerr <<  "open error!" << endl;
		exit(-1);
	}
	int srcid,dstid;
	char ch;
	while(!fin.eof())
	{
		fin >> srcid >> ch >> dstid;
		if(fin.good())
		{
			if(part[srcid] != part[dstid])
				cut[0]++;
			bound[0]++;
		}
		else{
			fin.clear();
			break;
		}
	}
	fin.close();
	cut[0] -= (cut[1]+cut[2]);
	bound[0] -= (bound[1]+bound[2]);
	double cuttingRate = 100.00 * double(cut[0]) / double(bound[0]);
	cout << "cutting rate of other   edges:" << cuttingRate << "%" << endl;
}

int main()
{
	readRes();
	readImportantEdges();
	readCommonEdges();
	return 0;
}