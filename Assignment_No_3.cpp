/*
 * Problem Statement :-
    Implement Min, Max, Sum and Average operations using Parallel Reduction.
*/

#include<bits/stdc++.h>
#include<omp.h>
using namespace std;

void min_reduction(vector<int> &arr)
{
    int minn = INT_MAX;
    #pragma omp parallel for reduction(min: minn)
    for(int i=0; i<arr.size(); i++)
    {
        if(arr[i] < minn) minn = arr[i];
    }
    cout<<"\n\t Minimum Value: "<<minn<<endl;
}

void max_reduction(vector<int> &arr)
{
    int maxx = INT_MIN;
    #pragma omp parallel for reduction(max: maxx)
    for(int i=0; i<arr.size(); i++)
    {
        if(arr[i] > maxx) maxx = arr[i];
    }
    cout<<"\n\t Maximum Value: "<<maxx<<endl;
}

void sum_reduction(vector<int> &arr)
{
    int sum = 0;
    #pragma omp parallel for reduction(+: sum)
    for(int i=0; i<arr.size(); i++)
    {
        sum += arr[i];
    }
    cout<<"\n\t Sum: "<<sum<<endl;
}

void average_reduction(vector<int> &arr)
{
    int sum = 0;
    #pragma omp parallel for reduction(+: sum)
    for(int i=0; i<arr.size(); i++)
    {
        sum += arr[i];
    }
    cout<<"\n\t Average: "<<sum/(double)arr.size()<<endl;
}

int main() 
{
    int n;
    cout<<"\n\t Enter size of the Array : ";
    cin>>n;
    vector<int> a(n);
    cout<<"\n\t Enter Array: ";
    for(int i=0; i<n; i++) cin>>a[i];  

    min_reduction(a);
    max_reduction(a);
    sum_reduction(a);
    average_reduction(a);
}