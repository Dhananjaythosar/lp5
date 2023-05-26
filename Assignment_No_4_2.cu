/*
 * Problem Statement :-
    Write a CUDA Program using CUDA C for :
        1. Addition of two large vectors
        2. Matrix Multiplication 
*/

//2. Matrix Multiplication 

#include <iostream>
#include <cuda_runtime.h>

#define N 1024
#define THREADS_PER_BLOCK 32


// The blockIdx variable contains the index of the current block
// threadIdx contains the index of the current thread within its block
// The blockDim variable contains the dimensions of the block
// blockIdx.x and blockIdx.y are the x and y indices of the block 
// threadIdx.x and threadIdx.y are the x and y indices of the thread within the block
__global__ void matrix_multiply(int *a, int *b, int *c, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) 
    {
        int sum = 0;
        for (int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

int main()
{
    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;
    int size = N * N * sizeof(int);

    // Allocate memory on host
    h_a = (int *)malloc(size);
    h_b = (int *)malloc(size);
    h_c = (int *)malloc(size);

    // Initialize matrices
    for (int i = 0; i < N * N; i++)
    {
        h_a[i] = 1;
        h_b[i] = 2;
        h_c[i] = 0;
    }

    // Allocate memory on device
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy input data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch kernel on device
    dim3 grid((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1);
    dim3 block(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    matrix_multiply<<<grid, block>>>(d_a, d_b, d_c, N);

    // Copy output data from device to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify the result
    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            printf("%d ", h_c[i * N + j]); 
        }
        printf("\n");
    }

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free memory on host
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}