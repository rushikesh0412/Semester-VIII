
//Parallel Breadth First Search (BFS) Traversal Code:
#include <bits/stdc++.h>
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




/*

g++ -fopenmp HPC_Assg_01.cpp -o HPC_Assg_01
./HPC_Assg_01

8 
7
1 2
1 3
2 4
2 5
3 6
3 7
1 6
1


BFS Traversal: 0 1 1 2 2 2 2 
Parallel BFS took 0.0021291 seconds to execute


*/

/*
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
       
	}

}
*/
/*
g++ -fopenmp HPC_Assg_01.cpp -o HPC_Assg_01
./HPC_Assg_01

7 6
1 2
1 3
2 4
2 5
3 6
3 7
1



Parallel DFS Traversal: 0 2 4 5 1 3 6 7 
Parallel DFS took 0.0016511 seconds to execute

*/

