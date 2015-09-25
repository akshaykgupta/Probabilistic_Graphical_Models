#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <set>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <random>
#include <chrono>
#include <string>

using namespace std;

typedef vector< pair< vector<int>, vector<int> > > Graph;   //Each node has a parent vector and a child vector

class BayesNet {
	Graph Network;
	int N;
	
	public:
		void generateNetwork(int, int);   //Generate a random network
		void readNetwork(string);         //Read a network from a file
		void writeNetwork(string);        //Write a network to a file
		void outputNetwork();             //Write a network to std output
		void findNodalDSeps(string, string);    //Reads query file, computes d-sep nodes, writes to output file 
		void bayesBall(vector<bool>&, vector<int>&, vector<int>&, vector<int>&, vector<int>&, vector<int>&);  //Implements Bayes' Ball Algorithm
		void findAllDSeps(string, string);   //Find all pairs of d-sep nodes
		void readQuery(string, vector<bool>&, vector<int>&);   //Read a query
		void readQuery(string, vector<bool>&); //Read a query (extra credit Q)
		void writeQuery(ofstream&, vector<bool>&, vector<int>&, vector<int>&, vector<int>&, vector<int>&); //write result of a query
		void writeQuery(ofstream&, vector< vector<bool> >&);  //write result of a query (extra credit Q)
};

void BayesNet::generateNetwork(int n, int k)
{
	//Generate a random network
	
	N = n;
	Network.clear();
	Network.resize(N);
	vector<int> numList(N - 1);
	vector<int> shuffleList;
	for(int i = 0; i < N - 1; i++)
		numList[i] = N - 1 - i;
	int u, v;
	for(int i = 0; i < N; i++)     //For each node in topological order
	{
		u = (rand() % k) + 1;      //Choose u (number of children)
		
		if(u < (N - i - 1))   //If more than u candidates exist for children
		{
			shuffleList.resize(N - i - 1);
			copy(numList.begin(), numList.end(), shuffleList.begin());
		    unsigned seed = chrono::system_clock::now().time_since_epoch().count();    //Choose a time-based seed
			//unsigned seed = 200;
			shuffle(shuffleList.begin(), shuffleList.end(), default_random_engine(seed));  //Uniformly shuffle the list (i+1 ... n)
			
			//The first u elements of shuffled list are children of ith node
			//Add edges to the network
			for(int j = 0; j < u; j++)
			{
				Network[i].first.push_back(shuffleList[j]);
				Network[shuffleList[j]].second.push_back(i);
			}
			numList.pop_back();
		}
		
		else      //At most u candidates exist for children
		{
			for(int j = i + 1; j < N; j++)
			{
				Network[i].first.push_back(j);
				Network[j].second.push_back(i);
			}
		}
	}
}

void BayesNet::readNetwork(string filename)
{
	// Takes in a file name and reads in a Bayesian network from that file
	
	ifstream f(filename);
	f >> N;
	Network.clear();
	Network.resize(N);
	int node, child;
	string line, temp;
	for(int i = 0; i < N; i++)
	{
		f >> node;
		node -= 1;
		f >> line;
		for(int j = 1; j < ((int) line.size()); j++)
		{
			if(line[j] != ' ' && line[j] != ',' && line[j] != ']')
			{
				temp.push_back(line[j]);
			}
			else if(!temp.empty())
			{
				child = stoi(temp) - 1;
				Network[node].first.push_back(child);
				Network[child].second.push_back(node);
				temp.clear();
			}
		}
	}
	f.close();
}

void BayesNet::writeNetwork(string filename)
{
	//Takes a file name and writes the Bayesian network to the file
	
	ofstream f(filename);
	f << N << "\n";
	for(int i = 0; i < N; i++)
	{
		f << i + 1;
		f << " [";
		int len = (int) Network[i].first.size();
		for(int j = 0; j < len; j++)
		{
			f << Network[i].first.at(j) + 1;
			if(j < len - 1)
				f << ",";
		}
		f << "]\n";
	}
	f.close();
}

void BayesNet::outputNetwork()
{
	//Writes the Bayesian network to the command line
	
	cout << N << "\n";
	for(int i = 0; i < N; i++)
	{
		cout << i + 1;
		cout << " [";
		int len = (int) Network[i].first.size();
		for(int j = 0; j < len; j++)
		{
			cout << Network[i].first.at(j) + 1;
			if(j < len - 1)
				cout << ",";
		}
		cout << "]\n";
	}
}

void BayesNet::findNodalDSeps(string queryFile, string outputFile)
{
	//Reads query from file; computes d-separeated, req. probalility and req. observation nodes for each query
	
	ifstream fin(queryFile);
	ofstream fout(outputFile);
	string line;
	int q, i = 0;
	getline(fin, line);
	q = stoi(line);
	while(getline(fin, line) && i < q)    //read query line by line
	{
		vector<int> queryNodes;
		vector<bool> observed(N, false);
		readQuery(line, observed, queryNodes);   //read current query
		vector<int> irrelevant, relevant, reqProb, reqObs;
		bayesBall(observed, queryNodes, irrelevant, reqProb, reqObs, relevant);   //run Bayes' ball on observed and query nodes
		writeQuery(fout, observed, queryNodes, irrelevant, reqProb, reqObs);   //write the result to file
		i++;
	}
	fin.close();
	fout.close();
}

void BayesNet::bayesBall(vector<bool>& observed, vector<int>& queryNodes, vector<int>& irrelevant, vector<int>& reqProb, vector<int>& reqObs, vector<int>& relevant)
{
	//The Bayes' Ball algorithm
	
	vector< pair<int, int> > frontier;   //list of nodes to be visited. Each node is a pair denoting node no. and whether it was visited by a child or parent
	
	vector<bool> visited(N), topMarked(N), bottomMarked(N);
	
	for(int i = 0; i < (int) queryNodes.size(); i++)
		frontier.push_back(make_pair(queryNodes[i], 2));    //initialise frontier with the query nodes
	
	while(!frontier.empty())
	{
		pair<int, int> nextNode = frontier.back();    //pop the next node
		frontier.pop_back();
		visited[nextNode.first] = true;      //mark it as visited
		if(nextNode.second == 1)      //if node is visited by a parent
		{
			if(!observed[nextNode.first] && !bottomMarked[nextNode.first])   // if node is not observed and not marked on bottom
			{
				bottomMarked[nextNode.first] = true;    //mark node on bottom
				int len = (int) Network[nextNode.first].first.size();
				for(int i = 0; i < len; i++)    //schedule its children to be visited
					frontier.push_back(make_pair(Network[nextNode.first].first.at(i), 1));   
			}
			else if(observed[nextNode.first] && !topMarked[nextNode.first])  //if node is observed and not marked on top
			{
				topMarked[nextNode.first] = true;    //mark node on top
				int len = (int) Network[nextNode.first].second.size();
				for(int i = 0; i < len; i++)     //schedule its parents to be visited
					frontier.push_back(make_pair(Network[nextNode.first].second.at(i), 2));
			}
		}
		else if(!observed[nextNode.first])     //is node is visited by child and is not observed
		{
			if(!bottomMarked[nextNode.first])    //if node is not marked on bottom
			{
				bottomMarked[nextNode.first] = true;     //mark node on bottom
				int len = (int) Network[nextNode.first].first.size();
				for(int i = 0; i < len; i++)    //schedule its children to be visited
					frontier.push_back(make_pair(Network[nextNode.first].first.at(i), 1));
			}
			if(!topMarked[nextNode.first])      //if node is not marked on top
			{
				topMarked[nextNode.first] = true;     //mark not on top
				int len = (int) Network[nextNode.first].second.size();
				for(int i = 0; i < len; i++)      //schedule its parents to be visited
					frontier.push_back(make_pair(Network[nextNode.first].second.at(i), 2));
			}
		}
	}
	for(int i = 0; i < N; i++)
	{
		if(!bottomMarked[i])
			irrelevant.push_back(i);          //irrelevant nodes
		else
			relevant.push_back(i);            //relevant nodes
		if(topMarked[i])
			reqProb.push_back(i);             //requisite probability nodes
		if(visited[i] && observed[i])
			reqObs.push_back(i);              //requisite observation nodes
	}
}

void BayesNet::findAllDSeps(string queryFile, string outputFile)
{
	//Extra credit Q. Find all pairs of d-separated nodes
	
	ifstream fin(queryFile);
	ofstream fout(outputFile);
	string line;
	fin >> line;
	vector<bool> observed(N, false);
	readQuery(line, observed);        //read the observed nodes
	vector<bool> marked(N, false);
	vector< vector<bool> > connected(N, vector<bool>(N, false));   //nxn matrix : Aij = true => ith and jth node are d-connected
	
	for(int i = 0; i < N; i++)
	{
		if(observed[i] || marked[i])        //No need to run Bayes' ball on nodes that are observed or are a child of an unobserved node
			continue;
		marked[i] = true;  //set current node as marked
		vector<int> queryNodes;
		queryNodes.push_back(i);      //Query node is current (ith) node
		vector<int> irrelevant, relevant, reqObs, reqProb;
		bayesBall(observed, queryNodes, irrelevant, reqProb, reqObs, relevant);   //Run Bayes' ball on query and observed nodes
		
		//Nodes in vector 'relevant' are all nodes that were reached from i. All these nodes are d-connected to each other.
		for(int j = 0; j < (int) relevant.size(); j++)
		{
			for(int k = j + 1; k < (int) relevant.size(); k++)
			{
				connected[relevant[j]][relevant[k]] = true;      
				connected[relevant[k]][relevant[j]] = true;
			}
			
			//ith node is d-connected to nodes in 'relevant'
			connected[i][relevant[j]] = true;   
			connected[relevant[j]][i] = true;
			
			//Set all nodes in relevant to 'marked'
			marked[relevant[j]] = true;
		}
	}
	writeQuery(fout, connected);     //write the result to file
	fin.close();
	fout.close();
}

void BayesNet::readQuery(string line, vector<bool>& observed, vector<int>& queryNodes)
{
	//reads a query from file
	
	string temp;
	bool flag = false;
	for(int j = 1; j < ((int) line.size()); j++)
	{
		if(line[j] == '[')
			flag = true;
		else if(line[j] == ' ')
		{}
		else if(line[j] != ',' && line[j] != ']')
			temp.push_back(line[j]);
		else if(!temp.empty())
		{
			int node = stoi(temp) - 1;
			if(!flag)
				queryNodes.push_back(node);
			else
				observed[node] = true;
			temp.clear();
		}
	}
}

void BayesNet::readQuery(string line, vector<bool>& observed)
{
	//reads observed nodes from file (Extra credit Q)
	
	string temp;
	for(int j = 1; j < ((int) line.size()); j++)
	{
		if(line[j] != ' ' && line[j] != ',' && line[j] != ']')
		{
			temp.push_back(line[j]);
		}
		else if(line[j] != ' ' && !temp.empty())
		{
			int node = stoi(temp) - 1;
			observed[node] = true;
			temp.clear();
		}
	}
}

void BayesNet::writeQuery(ofstream& fout, vector<bool>& observed, vector<int>& queryNodes, vector<int>& irrelevant, vector<int>& reqProb, vector<int>& reqObs)
{
	//Writes result of a query to file
	
	fout << "query:[";
	for(int i = 0; i < (int) queryNodes.size(); i++)
	{
		fout << queryNodes[i] + 1;
		if(i < (int) queryNodes.size() - 1)
			fout << ",";
	}
	fout << "] obs:[";
	int commas = 0;
	for(int i = 0; i < N; i++)
	{
		if(observed[i])
		{
			if(commas != 0)
				fout << ",";
			fout << i + 1;
			commas++;
		}
	}
	fout << "] dsep:[";
	for(int i = 0; i < (int) irrelevant.size(); i++)
	{
		fout << irrelevant[i] + 1;
		if(i < (int) irrelevant.size() - 1)
			fout << ",";
	}
	fout << "] req-prob:[";
	for(int i = 0; i < (int) reqProb.size(); i++)
	{
		fout << reqProb[i] + 1;
		if(i < (int) reqProb.size() - 1)
			fout << ",";
	}
	fout << "] req-obs:[";
	for(int i = 0; i < (int) reqObs.size(); i++)
	{
		fout << reqObs[i] + 1;
		if(i < (int) reqObs.size() - 1)
			fout << ",";
	}
	fout <<"]\n";
}

void BayesNet::writeQuery(ofstream& fout, vector < vector<bool> >& connected)
{
	//Writes all pairs of d-sep nodes to file (Extra credit Q)
	
	for(int i = 0; i < N; i++)
	{
		for(int j = i + 1; j < N; j++)
		{
			if(!connected[i][j])
				fout << "[" << i + 1 << "," << j + 1 << "]\n";
		}
	}
}

int main(int argc, char* argv[]) {
	srand(time(NULL));
	if(argc < 2)
		cout << "Enter command line arguments.\n";
	else if(string(argv[1]) == "Generate")
	{
		if(argc < 5)
			cout << "Not enough arguments";
		else
		{
			int n = atoi(argv[2]);
			int k = atoi(argv[3]);
			BayesNet net;
			net.generateNetwork(n, k);
			string filename(argv[4]);
			net.writeNetwork(filename);
		}
	}
	else if (string(argv[1]) == "Query")
	{
		if(argc < 5)
			cout << "Not enough arguments";
		else
		{
			string infile(argv[2]);
			string queryfile(argv[3]);
			string outputfile(argv[4]);
			BayesNet net;
			net.readNetwork(infile);
			net.findNodalDSeps(queryfile, outputfile);
		}
	}
	else if (string(argv[1]) == "EQ")
	{
		if(argc < 5)
			cout << "Not enough arguments";
		else
		{
			string infile(argv[2]);
			string queryfile(argv[3]);
			string outputfile(argv[4]);
			BayesNet net;
			net.readNetwork(infile);
			net.findAllDSeps(queryfile, outputfile);
		}
	}
	else
		cout << "Wrong argument.\n";
}
