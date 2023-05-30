/*
/#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

void parallelBFS(vector<int> adj[], int V, int source)
{
    int level[V];
    bool visited[V];
    memset(visited, false, sizeof visited);
    visited[source] = true;
    level[source] = 0;
    queue<int> Q;
    Q.push(source);
    while (!Q.empty())
    {
        int u = Q.front();
        Q.pop();
        #pragma omp parallel for
        for (int i = 0; i < adj[u].size(); i++)
        {
            int v = adj[u][i];
            if (!visited[v])
            {
                visited[v] = true;
                level[v] = level[u] + 1;
                Q.push(v);
            }
        }
    }
    cout << "BFS Traversal: ";
    for (int i = 0; i < V; i++)
        cout << level[i] << " ";
}

int main()
{
    int V, E;
    cout << "Enter the number of vertices: ";
    cin >> V;
    cout << "Enter the number of edges: ";
    cin >> E;
    vector<int> adj[V];
    int u, v;
    for (int i = 0; i < E; i++)
    {
        cout << "Enter edge " << i + 1 << ": ";
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    
    int source;
    cout << "Enter the source vertex: ";
    cin >> source;
    double start_time = omp_get_wtime();
    parallelBFS(adj, V, source);
    double end_time = omp_get_wtime();
    cout << "\nParallel BFS took " << end_time - start_time << " seconds to execute\n";
    return 0;
}

*/





/*

g++ -fopenmp HPC_Assg_01.cpp -o HPC_Assg_01
./HPC_Assg_01

Enter the number of vertices: 6
Enter the number of edges: 7
Enter edge 1: 0 1
Enter edge 2: 0 2
Enter edge 3: 1 3
Enter edge 4: 2 4
Enter edge 5: 3 4
Enter edge 6: 3 5
Enter edge 7: 4 5
Enter the source vertex: 0



BFS Traversal: 0 1 2 3 4 5
Parallel BFS took 0.000270264 seconds to execute



*/


//Parallel Depth First Search (DFS) Traversal Code:
#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

void parallelDFS(vector<int> adj[], int u, bool visited[], int level)
{
    visited[u] = true;
    cout << u << " ";
    #pragma omp parallel for
    for (int i = 0; i < adj[u].size(); i++)
    {
        int v = adj[u][i];
        if (!visited[v])
            parallelDFS(adj, v, visited, level + 1);
    }
}

void DFS(vector<int> adj[], int V, int source)
{
    bool visited[V];
    memset(visited, false, sizeof visited);
    double start_time = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < V; i++)
    {
        if (!visited[i])
            parallelDFS(adj, i, visited, 0);
    }
    double end_time = omp_get_wtime();
    cout << "\nParallel DFS took " << end_time - start_time << " seconds to execute\n";
}

int main()
{
    int V, E;
    cout << "Enter the number of vertices: ";
    cin >> V;
    cout << "Enter the number of edges: ";
    cin >> E;
    vector<int> adj[V];
    int u, v;
    for (int i = 0; i < E; i++)
    {
        cout << "Enter edge " << i + 1 << ": ";
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u); // Assuming an undirected graph
    }

    DFS(adj, V, 0); // Perform DFS traversal from vertex 0

    return 0;
}






/*
g++ -fopenmp HPC_Assg_01.cpp -o HPC_Assg_01
./HPC_Assg_01

Enter the number of vertices: 6
Enter the number of edges: 7
Enter edge 1: 0 1
Enter edge 2: 0 2
Enter edge 3: 1 3
Enter edge 4: 2 4
Enter edge 5: 3 4
Enter edge 6: 3 5
Enter edge 7: 4 5



0 1 3 4 2 5 
Parallel DFS took 0.000303671 seconds to execute


*/

