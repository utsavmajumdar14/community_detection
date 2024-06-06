#include <iostream>
#include <vector>
#include <unordered_map>
#include <random>
#include <cmath>
#include <bits/stdc++.h>
#include <fstream>
#include <sstream>

using namespace std;

void MemLPA(vector<vector<float>>adjacency_matrix)
{
    int n_nodes = adjacency_matrix.size();
    vector<int> labels(n_nodes,0); // optimize to direct initialization
    unordered_map<int, unordered_map<int, float>> mem;  // Is this the best data structure?

    for(int i=0; i<n_nodes; i++)
    {
        labels[i]=i;
        mem[i]={};
    }

    vector<bool> AL(n_nodes,0);  // Active List Check

    vector<vector<float>> label_history(4, vector<float>(n_nodes, 0.0)); // label history

    default_random_engine generator;
    normal_distribution<double> distribution(0.0, 1.0);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < n_nodes; j++)
            label_history[i][j] = distribution(generator);

    int n_iteration = 0;

    unordered_map<int, unordered_map<int, float>> global_neighbor_labels;

    while (find(AL.begin(), AL.end(), false) != AL.end() && n_iteration<=1000)
    {
        //cout<<"\nIter : "<<n_iteration<<"\n";
        for(int node=0; node < n_nodes; node++)
        {
            if(!AL[node])
            {
                unordered_map<int, float> neighbor_labels = global_neighbor_labels[node];

                for (int neighbor = 0; neighbor < n_nodes; neighbor++)

                {
                    float edge = adjacency_matrix[node][neighbor];
                    if (edge != 0)
                    {
                        neighbor_labels[labels[neighbor]] += edge;
                    }
                }
                int max_key = -1;
                float max_value = numeric_limits<float>::lowest();
                for (const auto& kv : neighbor_labels)
                {
                    if (kv.second > max_value)
                    {
                        max_key = kv.first;
                        max_value = kv.second;
                    }
                }
                labels[node] = max_key;
            }
        }
        int index = n_iteration % 4;
        for (int j=0; j<n_nodes; j++)
            label_history[index][j] = labels[j];

        for (int j=0; j<n_nodes; j++)
        {
            if(label_history[0][j]==label_history[1][j])
                if(label_history[1][j]==label_history[2][j])
                    if(label_history[2][j]==label_history[3][j])
                        AL[j]=1;
        }
        ofstream f2;
        f2.open("Seq_LFRout_10000_0.4.csv");
        int counter = 1;
        for (auto x : labels)
            f2 << counter++ << "," << x << endl;

        f2.close();
        n_iteration++;
    }
}

int main()
{

    clock_t tStart = clock();
    ifstream f;
    f.open("LFR-10000-0.4.txt");
    int nodes = 10000;
    vector<vector<float>> adjacency_matrix(nodes, vector<float>(nodes,0.0));
    string line;
    while(getline(f, line))
    {
        stringstream ss(line);
        int n1, n2;
        ss >> n1 >> n2;

        adjacency_matrix[n1][n2] = 1.0;
        adjacency_matrix[n2][n1] = 1.0;
    }

    f.close();
    vector<vector<int>> neigh_count(nodes, vector<int>(nodes));
    vector<vector<int>> int_neigh_count(nodes, vector<int>(nodes));

    vector<int> all_neigh_count(nodes);
    vector<vector<float>> wtd_neigh_count(nodes, vector<float>(nodes));
    //vector<vector<float>> added_neigh_count(nodes, vector<float>(nodes));
    vector<vector<float>> prod_neigh_count(nodes, vector<float>(nodes));

    for(int i=0; i<nodes; i++)
    {
        int c=0;
        for(int j=0; j<nodes; j++)
            if(adjacency_matrix[i][j]!=0)
                c++;
        all_neigh_count[i]=c;
    }


    for(int i=0; i<nodes; i++)
        for(int j=0; j<nodes; j++)
            if(j!=i)
            {
                int c=0;
                for(int k=0; k<nodes; k++)
                    if(adjacency_matrix[i][k] != 0 && adjacency_matrix[j][k] !=0 && adjacency_matrix[i][j] != 0)
                        int_neigh_count[i][j] = ++c;
            }
     for(int i=0; i<nodes; i++)
        for(int j=0; j<nodes; j++)
            if(adjacency_matrix[i][j] > 0)
                neigh_count[i][j] = int_neigh_count[i][j] + 1;
            else
                neigh_count[i][j] = int_neigh_count[i][j];

    for(int i=0; i<nodes; i++)
        for(int j=0; j<nodes; j++)
            wtd_neigh_count[i][j] = neigh_count[i][j]*1.0/all_neigh_count[i];

    for(int i=0; i<nodes; i++)
        for(int j=0; j<nodes; j++)
        {
            //added_neigh_count[i][j] = adjacency_matrix[i][j]*1.0 + wtd_neigh_count[i][j];
            prod_neigh_count[i][j] = adjacency_matrix[i][j]*1.0 * wtd_neigh_count[i][j];
        }
    cout<<"\n\n* * * Execution using product-weighted matrix * * *\n\n";
    MemLPA(prod_neigh_count);
    printf("\nTime taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);

    return 0;
}
