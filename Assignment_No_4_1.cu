/*
 * Problem Statement :-
    Write a CUDA Program using CUDA C for :
        1. Addition of two large vectors
        2. Matrix Multiplication 
*/

//1. Addition of two large vectors

#include <iostream> 
#include <cuda_runtime.h>

#define N 100000
#define THREADS_PER_BLOCK 1024

__global__ void add(int *a, int *b, int *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        c[i] = a[i] + b[i];
}

int main()
{
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size = N * sizeof(int);

    // Allocate memory on host
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    // Initialize arrays
    for (int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = i * 2;
        c[i] = 0;
    }

    // Allocate memory on device
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy input data from host to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch kernel on device
    // grid represent => size of the grid of blocks that will be launched on the device(GPU)
    // 1st dimension => number of blocks required to launch N threads with THREADS_PER_BLOCK threads per block
    // 2nd and 3rd dimension => 1 since we are launching a one-dimensional grid
    dim3 grid((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, 1);
    // The block variable is used to specify the size of each block
    dim3 block(THREADS_PER_BLOCK, 1, 1);
    // Each thread in the grid will execute the kernel function
    add<<<grid, block>>>(d_a, d_b, d_c);

    // Copy output data from device to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Display the result
    for (int i = 0; i < N; i++)
    {
        printf("%d ", c[i]);
    }

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free memory on host
    free(a);
    free(b);
    free(c);

    return 0;
}