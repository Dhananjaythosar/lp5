/*
 * Problem Statement :-
    Design and implement Parallel Breadth First Search and Depth First Search based on existing algorithms
    using OpenMP. Use a Tree or an undirected graph for BFS and DFS.
*/

#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

vector<vector<int>> AdjList;
vector<bool> visited;

void displayAdjList()
{
    cout<<"\n\t Adjacency List : "<<endl;
    int current = 0;
    for(auto itr:AdjList)
    {
        cout<<"\n\t "<<current<<" ===> ";
        for(auto itr2: itr)
        {
            cout<<itr2<<" ";
        }
        cout<<endl;
        current++;
    }
}

void parallelBFS(int start)
{
    queue<int> q;
    visited[start] = true;
    q.push(start);

    while(!q.empty())
    {
        int current = q.front();
        q.pop();
        cout<<current<<" ";

        #pragma parallel for
        for(int i=0; i<AdjList[current].size(); i++)
        {
            int neighbor = AdjList[current][i];
            if(!visited[neighbor])
            {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
}

void parallelDFS(int start)
{
    stack<int> s;
    s.push(start);
    visited[start] = 1;

    while (!s.empty()) 
    {
        int current = s.top();
        s.pop();
        cout<<current<< " ";

		#pragma omp parallel for
        for(int i=0; i<AdjList[current].size(); i++)
        {
            int neighbor = AdjList[current][i];
            if(!visited[neighbor])
            {
                visited[neighbor] = true;
                s.push(neighbor);
            }
        }
    }
}

int main()
{
    int vertices, edges, start, choice;

    cout<<"\n\t Enter number of vertices and edges : ";
    cin>>vertices>>edges;
    AdjList.assign(vertices, vector<int>());

    for(int i=0; i<edges; i++)
    {
        int source, destination;
		cout<<"\n\t Enter "<<i+1<<" edge"<<endl;
		cout<<"\n\t Enter source vertex : ";
		cin>>source;
		cout<<"\n\t Enter destination vertex : ";
		cin>>destination;
        AdjList[source].push_back(destination);
        AdjList[destination].push_back(source);
    }

    while(true)
    {
        cout<<"\n\t ===Menu=== \n\t\t 1. DFS Traversal \n\t\t 2. BFS Traversal \n\t\t 3. Display Adjacency list \n\t\t 4. Exit"<<endl;
        cout<<"\n\t Enter choice : ";
        cin>>choice;
        if(choice == 1)
        {
            cout<<"\n\t Enter starting vertex : ";
            cin>>start;
            visited.assign(vertices, false);
            cout<<"\n\t DFS Traversal: ";
            parallelDFS(start);
        }
        else if(choice == 2)
        {
            cout<<"\n\t Enter starting vertex : ";
            cin>>start;
            visited.assign(vertices, false);
            cout<<"\n\t BFS Traversal: ";
            parallelBFS(start);
        }
        else if(choice == 3)
        {
            displayAdjList();
        }
        else if(choice == 4)
        {
            exit(0);
        }
        else
        {
            cout<<"\n\t Invalid choice..!!"<<endl;
        }
    }
}