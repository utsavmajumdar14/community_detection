int main()
{

    int nodes = 10;
    vector<vector<float>> adjacency_matrix =  {{0, 1.0, 0.7, 0, 0, 0, 0, 0, 0, 0},
                                                {1.0, 0, 0.8, 0, 0, 0, 0.2, 0, 0, 0},
                                                {0.7, 0.8, 0, 0.3, 0, 0, 0, 0, 0, 0},
                                                {0, 0, 0.3, 0, 0.7, 0.8, 0, 0, 0, 0},
                                                {0, 0, 0, 0.7, 0, 0.9, 0, 0, 0, 0},
                                                {0, 0, 0, 0.8, 0.9, 0, 0, 0, 0, 0.3},
                                                {0, 0.2, 0, 0, 0, 0, 0, 0.9, 0.7, 0.8},
                                                {0, 0, 0, 0, 0, 0, 0.9, 0, 0.5, 0.6},
                                                {0, 0, 0, 0, 0, 0, 0.7, 0.5, 0, 0.4},
                                                {0, 0, 0, 0, 0, 0.3, 0.8, 0.6, 0.4, 0}};

    vector<int> all_neigh_count(nodes);
    vector<vector<float>> added_neigh_count(nodes, vector<float>(nodes));
    vector<vector<float>> prod_neigh_count(nodes, vector<float>(nodes));

    for (int i = 0; i < nodes; i++)
    {
        int c = 0;
        vector<int> int_neigh_count(nodes);
        for (int j = 0; j < nodes; j++)
            if (adjacency_matrix[i][j] != 0)
            {
                c++;
                int_neigh_count[j] = 1;
            }
        all_neigh_count[i] = c;
        for (int j = 0; j < nodes; j++)
            if (j != i && adjacency_matrix[i][j] != 0)
                for (int k = 0; k < nodes; k++)
                    if (adjacency_matrix[j][k] != 0 && int_neigh_count[k] != 0)
                        int_neigh_count[j] += 1;


        for (int j = 0; j < nodes; j++)
            if (i != j)
            {
                float wtd_neigh_count = int_neigh_count[j] * 1.0 / all_neigh_count[i];
                added_neigh_count[i][j] = adjacency_matrix[i][j] * 1.0 + wtd_neigh_count;
                prod_neigh_count[i][j] = adjacency_matrix[i][j] * 1.0 * wtd_neigh_count;
            }
    }

    cout << "\nAltered Added Neighbour Count : \n\n";
    for (int i = 0; i < nodes; i++)
    {
        for (int j = 0; j < nodes; j++)
            cout << added_neigh_count[i][j] << " ";
        cout << endl;
    }
    cout << endl;

    cout << "\nAltered Multiplied Neighbour Count : \n\n";
    for (int i = 0; i < nodes; i++) {
        for (int j = 0; j < nodes; j++) {
            cout << prod_neigh_count[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;

    cout<<"\n\n* * * Execution using vanilla matrix * * *\n\n";
    MemLPA(adjacency_matrix);
    cout<<"\n\n* * * Execution using added-weighted matrix * * *\n\n";
    MemLPA(added_neigh_count);
    cout<<"\n\n* * * Execution using product-weighted matrix * * *\n\n";
    MemLPA(prod_neigh_count);

    return 0;
}
