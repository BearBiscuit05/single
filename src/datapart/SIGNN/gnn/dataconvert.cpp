/* 将重要边权信息csv，转换为fennel图分区算法需要的输入格式 */
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
using namespace std;

/* 全图节点数 */
const int nodeNum = 2449029;
/* 重要边信息目标文件 */
const char* selectedEdgesFile = "/home/wsy/single-gnn/data/ogb-products/important-edges.csv";
/* 输出的fennel需要的输入格式文件 */
const char* graphFile = "/home/wsy/single-gnn/data/ogb-products/graph";
/* ogb-products数据集全图边信息 */
const char* allEdgesfile = "/home/wsy/single-gnn/data/ogb-products/edge.csv";

class edge
{
private:
	int srcid;
	int weight;
public:
	edge(const int sid,const int w)
	{
		srcid = sid;
		weight = w;
	}
	friend bool operator == (const edge& e1,const edge& e2);
	friend bool operator < (const edge& e1,const edge& e2);
	friend ostream& operator << (ostream& os,const edge& e); 
};

bool operator == (const edge& e1,const edge& e2){
	return e1.srcid == e2.srcid;
};
bool operator < (const edge& e1,const edge& e2){
	return e1.weight < e2.weight;
};
ostream& operator << (ostream& os,const edge& e){
	os << e.srcid << " " << e.weight << " ";
	return os;
};

set<edge>* edgeList;

// void getNodeNum()
// {
// 	cout << "输入数据集结点总数:" << endl;
// 	cin >> nodeNum;
// 	cout << "全图结点编号[0~" << nodeNum << ")" << endl;
// }

void initEdgeList()
{
	edgeList = new(nothrow) set<edge>[nodeNum];
	if(edgeList == NULL)
	{
		cerr << "new error!" << endl;
		exit(-1);
	}
}

void freeEdgeList()
{
	if(edgeList != NULL)
	{
		for(int i = 0;i < nodeNum;i++)
			edgeList[i].clear();
		delete[] edgeList;
	}
}

/* 读取有边权的重要边 */
void readEdges1()
{
	ifstream fin;
	fin.open(selectedEdgesFile,ios::in|ios::binary);
	if(!fin.is_open())
	{
		cerr <<  "open error!" << endl;
		freeEdgeList();
		exit(-1);
	}
	int srcid,dstid,weight;
	char ch1,ch2;
	while(!fin.eof())
	{
		fin >> srcid >> ch1 >> dstid >> ch2>> weight;
		edge e(srcid,weight);
		if(fin.good())
		{
			set<edge>::iterator it = edgeList[dstid].find(e);
			if(it != edgeList[dstid].end())
			{
				if(*it < e)
				{
					edgeList[dstid].erase(it);
					edgeList[dstid].insert(e);
				}
			}
			else
			{
				edgeList[dstid].insert(e);
			}
		}
		else{
			fin.clear();
			break;
		}
	}
	fin.close();
}

/* 读取没有边权的原文件，先读重要边 */
void readEdges2()
{
	ifstream fin;
	fin.open(allEdgesfile,ios::in|ios::binary);
	if(!fin.is_open())
	{
		cerr <<  "open error!" << endl;
		freeEdgeList();
		exit(-1);
	}
	int srcid,dstid;
	char ch;
	while(!fin.eof())
	{
		fin >> srcid >> ch >> dstid;
		edge e(srcid,1);//不重要边，默认边权1
		if(fin.good())
		{
			set<edge>::iterator it = edgeList[dstid].find(e);
			if(it == edgeList[dstid].end())//没找到，说明不是重要边
			{
				edgeList[dstid].insert(e);
			}
		}
		else{
			fin.clear();
			break;
		}
	}
	fin.close();
}

void genGraph()
{
	//文件格式，第一行结点总数
	//第二行开始，每一行第一个数表示第i号结点的边数，接下来是每一条边
	ofstream fout;
	fout.open(graphFile,ios::out|ios::binary);
	if(!fout.is_open())
	{
		freeEdgeList();
		exit(-1);
	}
	fout << nodeNum << endl;
	for(int i = 0;i < nodeNum;i++)
	{
		fout << edgeList[i].size() << " ";
		for(set<edge>::iterator it = edgeList[i].begin();it != edgeList[i].end();it++)
		{
			fout << *it;
		}
		fout << endl;
	}
	fout.close();
}

int main()
{
	initEdgeList();
	readEdges1();
	readEdges2();
	genGraph();
	freeEdgeList();
	return 0;
}