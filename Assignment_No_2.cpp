/*
 * Problem Statement :-
    Write a program to implement Parallel Bubble Sort and Merge sort using 
    OpenMP. Use existing algorithms and measure the performance of sequential 
    and parallel algorithms.
*/
#include<bits/stdc++.h>
#include<omp.h>
using namespace std;

void sequential_bubbleSort(vector<int> &a)
{
    int n = a.size();
    for (int i=0; i<n; i++) 
    {
        int start = i%2;
        for (int j=start; j<n-1; j+=2) 
        {
            if (a[j] > a[j+1]) 
            {
                swap(a[j], a[j + 1]);
            }
        }
    }
}

void parallel_bubbleSort(vector<int> &a)
{
    int n = a.size();
    for (int i=0; i<n; i++) 
    {
        int start = i%2;
        #pragma omp parallel for shared(a, start)
        for (int j=start; j<n-1; j+=2) 
        {
            if (a[j] > a[j+1]) 
            {
                swap(a[j], a[j + 1]);
            }
        }
    }
}

void merge(vector<int> &arr, int l, int m, int r) 
{
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    vector<int> L(n1), R(n2);

    for (i = 0; i < n1; i++)
    {
        L[i] = arr[l + i];
    }  
    for (j = 0; j < n2; j++)
    {
        R[j] = arr[m + 1 + j];
    }
        
    i = 0;
    j = 0;
    k = l;
    while (i < n1 && j < n2) 
    {
        if (L[i] <= R[j]) 
        {
            arr[k] = L[i];
            i++;
        } 
        else 
        {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) 
    {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) 
    {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void sequential_mergeSort(vector<int> &arr, int left, int right) 
{
    if (left < right) 
    {
        int mid = left + (right - left) / 2;

        sequential_mergeSort(arr, left, mid);
        sequential_mergeSort(arr, mid+1, right);

        merge(arr, left, mid, right);
    }
}

void parallel_mergeSort(vector<int> &arr, int left, int right) 
{
    if (left < right) 
    {
        int mid = left + (right - left) / 2;

        #pragma omp parallel sections num_threads(2)
        {
            #pragma omp section
            {
                parallel_mergeSort(arr, left, mid);
            }
            #pragma omp section
            {
                parallel_mergeSort(arr, mid+1, right);
            }
        }

        merge(arr, left, mid, right);
    }
}


void printArray(vector<int> &arr) 
{
    cout<<"\n\t Sorted Array: ";
    for (int i : arr) 
    {
        cout << i << " ";
    }
    cout << endl;
}

int main()
{
    int n, choice;
    double start_time, end_time;

    while(true)
    {
        cout<<"\n\t ===Menu=== \n\t\t 1. Bubble Sort \n\t\t 2. Merge Sort \n\t\t 3. Exit"<<endl;
        cout<<"\n\t Enter choice : ";
        cin>>choice;
        if(choice == 1)
        {
            cout<<"\n\t Enter size of the Array : ";
            cin>>n;
            vector<int> a(n);
            cout<<"\n\t Enter Array: ";
            for(int i=0; i<n; i++) cin>>a[i];

            vector<int> b = a;
            vector<int> c = a;

            start_time = omp_get_wtime(); 
            sequential_bubbleSort(b);
            end_time = omp_get_wtime(); 
            cout << "\n\t Time taken by the sequential algorithm: " << end_time - start_time << " seconds" << endl;
            printArray(b);

            start_time = omp_get_wtime(); 
            parallel_bubbleSort(c);
            end_time = omp_get_wtime(); 
            cout << "\n\t Time taken by the parallel algorithm: " << end_time - start_time << " seconds" << endl;
            printArray(c);
        }
        else if(choice == 2)
        {
            cout<<"\n\t Enter size of the Array : ";
            cin>>n;
            vector<int> a(n);
            cout<<"\n\t Enter Array: ";
            for(int i=0; i<n; i++) cin>>a[i];

            vector<int> b = a;
            vector<int> c = a;

            start_time = omp_get_wtime(); 
            sequential_mergeSort(b, 0, n-1);
            end_time = omp_get_wtime(); 
            cout << "\n\t Time taken by the sequential algorithm: " << end_time - start_time << " seconds" << endl;
            printArray(b);

            start_time = omp_get_wtime(); 
            parallel_mergeSort(c, 0, n-1);
            end_time = omp_get_wtime(); 
            cout << "\n\t Time taken by the parallel algorithm: " << end_time - start_time << " seconds" << endl;
            printArray(c);
        }
        else if(choice == 3)
        {
            exit(0);
        }
        else
        {
            cout<<"\n\t Invalid choice..!!"<<endl;
        }
    }
    
}