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
	int bound[2] = {0,0};
	int cut[2] = {0,0};
	while(!fin.eof())
	{
		fin >> srcid >> ch1 >> dstid >> ch2 >> weight;
		if(fin.good())
		{
			int b = -1;
			if(weight == 50)
				b = 0;
			else if(weight == 20)
				b = 1;
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
		double(cut[0]) / double(bound[0]),
		double(cut[1]) / double(bound[1]),
	};
	cout << "cutting rate of bound 1 and 2:" << endl;
	cout << cuttingRate[0] << endl;
	cout << cuttingRate[1] << endl;
	cout << cut[0] << " " << cut[1] << endl;
	cout << bound[0] << " " << bound[1] << endl;
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
	int cut = 0;
	int all = 0;
	while(!fin.eof())
	{
		fin >> srcid >> ch >> dstid;
		if(fin.good())
		{
			if(part[srcid] != part[dstid])
				cut++;
			all++;
		}
		else{
			fin.clear();
			break;
		}
	}
	fin.close();
	double cuttingRate = double(cut) / double(all);
	cout << "average edges cutting rate:" << cuttingRate << endl;
}

int main()
{
	readRes();
	readImportantEdges();
	readCommonEdges();
	return 0;
}