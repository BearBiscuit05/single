#include <bits/stdc++.h>
#include <sys/types.h>
#include <unistd.h>
#define DB(x) cerr << __LINE__ << ": " << #x << " = " << (x) << endl
using namespace std;
using namespace std::chrono; 

// Number of partitions
const int k = 4;
const int mx = 2449030;
set<int> partitions[k];
set<int> adjacency[mx];
map<pair<int,int>,int> weight;

// Parameters
double alpha = 0.0000001;
double gammaPower = 0.99;

const char* inputFile = "../../data/ogb-products/graph";
const char* outputFile = "../../data/ogb-products/result";

double partitionCost(double sz) {
	double cost = 0;
	cost = alpha * pow(sz/1000+1.0, gammaPower) - alpha * pow(sz/1000, gammaPower);
	return cost;
}

double Cost(double sz) {
	double cost = 0;
	cost = alpha * pow(sz, gammaPower);
	return cost;
}

int main() {
	pid_t pid = getpid();
	cout << "ps -p " << pid << " -o pid,comm,rss" << endl;
	srand(42);
	ifstream fin;
	fin.open(inputFile,ios::in|ios::binary);
	if(!fin.is_open())
	{
		cout << "open error" << endl;
		exit(-1);
	}
	int n, m = 0;
	fin >> n;
	for(int i = 1; i <= n; ++i) {
		int number_of_nodes;
		fin >> number_of_nodes;
		m += number_of_nodes;
		for(int j = 0; j < number_of_nodes; ++j) {
			int x,y;
			fin >> x >> y;
			adjacency[i].insert(x);
			adjacency[x].insert(i);
			pair<int,int> edge(min(x,i),max(x,i));
			if(y == 1)
				y = -1600000;
			else if(y == 20)
				y = 20000;
			else if(y == 50)
				y = 10000;
			weight[edge] = y;
		}
	}
	//m /= 2;
	fin.close();
	cout << "already read all edges,start to run graph partition" << endl;
	
	// Results
	vector<int> cutEdge(k, 0);

	// Ordering of streaming vertices
	vector<int> ordering(n);
	for(int i = 1; i <= n; ++i) {
		ordering[i-1] = i;
	}
	random_shuffle(ordering.begin(), ordering.end());

	// Initial paritions
	for(int i = 0; i < k; ++i) {
		partitions[i].insert(ordering[i]);
	}

	auto start = high_resolution_clock::now(); 

	// Streaming vertices
	for(int node_number = k; node_number < n; ++node_number) {
		int finalPartition = 0, node = ordering[node_number], additionalEdge = 0;
		double objectiveFunctionScore = -1e18; 
		for(int container = 0; container < k; ++container) {
			double intraPartition = partitionCost((int)partitions[container].size());
			int interPartition = 0;
			for(auto neighbours: adjacency[node]) {
				if(partitions[container].find(neighbours) != partitions[container].end()) {
					pair<int,int> p(min(neighbours,node),max(neighbours,node));
					interPartition+=weight[p];
				}
			}
			// DB(interPartition);
			// DB(intraPartition);
			if(((double)interPartition) - intraPartition > objectiveFunctionScore) {
				objectiveFunctionScore = interPartition - intraPartition;
				finalPartition = container;
				additionalEdge = interPartition;
			}
		}
		partitions[finalPartition].insert(node);
		cutEdge[finalPartition] += additionalEdge;
	}

	cout << "complete graph partition,start to write partition file" << endl;
	ofstream fout;
	fout.open(outputFile,ios::out|ios::binary);
	if(!fout.is_open())
	{
		cout << "open error" << endl;
		exit(-1);
	}
	for(int i = 0; i < k; ++i) {
		fout << i+1 << " " << partitions[i].size() << "\n"; 
		for(auto it: partitions[i]) 
			fout << it << " ";
		fout << "\n";
	}
	fout.close();
	auto stop = high_resolution_clock::now(); 

	double result = 0;
	int totalCutEdges = 0;
	for(int i = 0; i < k; ++i) {
		DB(cutEdge[i]);
		DB(partitionCost(partitions[i].size()));
		result += ((double) cutEdge[i]) - Cost(partitions[i].size());
		totalCutEdges += cutEdge[i];
	}
	totalCutEdges = m - totalCutEdges;
	// DB(totalCutEdges);
	DB(result);
	auto duration = duration_cast<microseconds>(stop - start); 
	cout << "time cost of graph partition:" << duration.count()/1e6 << "s" << endl;

	return 0;
}