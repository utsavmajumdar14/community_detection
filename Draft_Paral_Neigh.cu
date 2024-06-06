#include <cuda.h>
#include <stdio.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

#define BLOCK_SIZE 16

__global__ void degreeKernel(float *d_adjacency_matrix, int *degrees, int nodes)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < nodes)
    {
        int degree = 0;
        for(int i = 0; i < nodes; i++) {
            degree += d_adjacency_matrix[tid * nodes + i] != 0;
        }
        degrees[tid] = degree;
    }
}

__global__ void common_neighbor_kernel(int nodes, float* adjacency_matrix, int* int_neigh_count)
{
    __shared__ float shared_matrix[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int shared_counts[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    shared_counts[ty][tx] = 0;

    if (row < nodes && col < nodes)
        shared_matrix[ty][tx] = adjacency_matrix[row * nodes + col];
    else
        shared_matrix[ty][tx] = 0;

    __syncthreads();

    if (row < nodes && col < nodes && row != col)
        for (int k = 0; k < BLOCK_SIZE; k++)
            if (shared_matrix[ty][k] != 0 && shared_matrix[k][tx] != 0 && shared_matrix[ty][tx] != 0)
                shared_counts[ty][tx]++;

    __syncthreads();

    if (row < nodes && col < nodes && row != col)
        atomicAdd(&int_neigh_count[row * nodes + col], shared_counts[ty][tx]);

}

__global__ void compute_neigh_count(float *d_adjacency_matrix, int *d_int_neigh_count, int *d_neigh_count, int nodes)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<nodes && j<nodes)
    {
        if(d_adjacency_matrix[i*nodes+j] > 0)
            d_neigh_count[i*nodes+j] = d_int_neigh_count[i*nodes+j] + 1;
        else
            d_neigh_count[i*nodes+j] = d_int_neigh_count[i*nodes+j];
    }
}

__global__ void compute_wtd_neigh_count(int *d_neigh_count, int *d_degrees, float *d_wtd_neigh_count, int nodes)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<nodes && j<nodes)
    {
        int index = i * nodes + j;
        d_wtd_neigh_count[index] = d_neigh_count[index]*1.0/d_degrees[i];
    }

}

__global__ void add_prod_kernel(float *d_adjacency_matrix, float *d_wtd_neigh_count, float *d_added_neigh_count, float *d_prod_neigh_count, int nodes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i<nodes && j<nodes)
    {
        int index = i * nodes + j;
        d_added_neigh_count[index] = d_adjacency_matrix[index] + d_wtd_neigh_count[index];
        d_prod_neigh_count[index] = d_adjacency_matrix[index] * d_wtd_neigh_count[index];
    }
}


int main()
{
    const int nodes = 10;
    const int adjacency_size = nodes * nodes;


    float h_adjacency_matrix[nodes][nodes] = {
        {0, 1.0, 0.7, 0, 0, 0, 0, 0, 0, 0},
        {1.0, 0, 0.8, 0, 0, 0, 0.2, 0, 0, 0},
        {0.7, 0.8, 0, 0.3, 0, 0, 0, 0, 0, 0},
        {0, 0, 0.3, 0, 0.7, 0.8, 0, 0, 0, 0},
        {0, 0, 0, 0.7, 0, 0.9, 0, 0, 0, 0},
        {0, 0, 0, 0.8, 0.9, 0, 0, 0, 0, 0.3},
        {0, 0.2, 0, 0, 0, 0, 0, 0.9, 0.7, 0.8},
        {0, 0, 0, 0, 0, 0, 0.9, 0, 0.5, 0.6},
        {0, 0, 0, 0, 0, 0, 0.7, 0.5, 0, 0.4},
        {0, 0, 0, 0, 0, 0.3, 0.8, 0.6, 0.4, 0}
    };

    float* d_adjacency_matrix;
    int* d_degrees;
    int* d_int_neigh_count;
    int *d_neigh_count;
    float *d_wtd_neigh_count;
    float *d_added_neigh_count;
    float *d_prod_neigh_count;



    cudaMalloc((void **)&d_adjacency_matrix, adjacency_size * sizeof(float));
    cudaMalloc((void **)&d_degrees, nodes * sizeof(int));
    cudaMalloc((void **)&d_int_neigh_count, adjacency_size * sizeof(int));
    cudaMalloc((void **)&d_neigh_count, adjacency_size * sizeof(int));
    cudaMalloc((void **)&d_wtd_neigh_count, adjacency_size*sizeof(float));
    cudaMalloc((void **)&d_added_neigh_count, adjacency_size*sizeof(float));
    cudaMalloc((void **)&d_prod_neigh_count, adjacency_size*sizeof(float));

    cudaMemset(d_int_neigh_count, 0, adjacency_size * sizeof(int));
    cudaMemset(d_neigh_count, 0, adjacency_size*sizeof(int));
    cudaMemset(d_wtd_neigh_count, 0.0, adjacency_size*sizeof(float));
    cudaMemset(d_added_neigh_count, 0.0, adjacency_size*sizeof(float));
    cudaMemset(d_prod_neigh_count, 0.0, adjacency_size*sizeof(float));

    cudaMemcpy(d_adjacency_matrix, h_adjacency_matrix, adjacency_size * sizeof(float), cudaMemcpyHostToDevice);

    //Launch Degree Kernel
    degreeKernel<<<(nodes + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_adjacency_matrix, d_degrees, nodes);

    int h_degrees[nodes];
    cudaMemcpy(h_degrees, d_degrees, nodes * sizeof(int), cudaMemcpyDeviceToHost);


    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((nodes + BLOCK_SIZE - 1) / BLOCK_SIZE, (nodes + BLOCK_SIZE - 1) / BLOCK_SIZE);

    common_neighbor_kernel<<<grid_size, block_size>>>(nodes, d_adjacency_matrix, d_int_neigh_count);

    cudaDeviceSynchronize();

    int h_int_neigh_count[nodes][nodes];
    cudaMemcpy(h_int_neigh_count, d_int_neigh_count, adjacency_size * sizeof(int), cudaMemcpyDeviceToHost);

    cout<<"\n Intermediate Neighbourhood Count : \n";
    for(int i=0; i<nodes; i++)
    {
        for(int j=0; j<nodes; j++)
            cout << h_int_neigh_count[i][j] << " ";
        cout << endl;
    }

    compute_neigh_count<<<grid_size, block_size>>>(d_adjacency_matrix, d_int_neigh_count, d_neigh_count, nodes);

    cudaDeviceSynchronize();

    int h_neigh_count[nodes][nodes];
    cudaMemcpy(h_neigh_count, d_neigh_count, adjacency_size*sizeof(int), cudaMemcpyDeviceToHost);

    cout<<"\n Neighbourhood Count : \n";
    for(int i=0; i<nodes; i++)
    {
        for(int j=0; j<nodes; j++)
            cout << h_neigh_count[i][j] << " ";
        cout << endl;
    }

    compute_wtd_neigh_count<<<grid_size, block_size>>>(d_neigh_count, d_degrees, d_wtd_neigh_count, nodes);

    cudaDeviceSynchronize();

    float h_wtd_neigh_count[nodes][nodes];
    cudaMemcpy(h_wtd_neigh_count, d_wtd_neigh_count, adjacency_size*sizeof(float), cudaMemcpyDeviceToHost);

    for(int i=0; i<nodes; i++)
        cout<<h_degrees[i]<<"\n";

    cout<<"\n Weighted Neighbourhood Count : \n";
    for(int i=0; i<nodes; i++)
    {
        for(int j=0; j<nodes; j++)
            cout << h_wtd_neigh_count[i][j] << " ";
        cout << endl;
    }

    add_prod_kernel<<<grid_size, block_size>>>(d_adjacency_matrix, d_wtd_neigh_count, d_added_neigh_count, d_prod_neigh_count, nodes);

    cudaDeviceSynchronize();

    float h_added_neigh_count[nodes][nodes];
    float h_prod_neigh_count[nodes][nodes];

    cudaMemcpy(h_added_neigh_count, d_added_neigh_count, adjacency_size*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_prod_neigh_count, d_prod_neigh_count, adjacency_size*sizeof(float), cudaMemcpyDeviceToHost);

    cout<<"\n Added Neighbourhood Count : \n";
    for(int i=0; i<nodes; i++)
    {
        for(int j=0; j<nodes; j++)
            cout << h_added_neigh_count[i][j] << " ";
        cout << endl;
    }

    cout<<"\n Product Neighbourhood Count : \n";
    for(int i=0; i<nodes; i++)
    {
        for(int j=0; j<nodes; j++)
            cout << h_prod_neigh_count[i][j] << " ";
        cout << endl;
    }

    cudaFree(d_degrees);
    cudaFree(d_adjacency_matrix);
    cudaFree(d_int_neigh_count);
    cudaFree(d_neigh_count);
    cudaFree(d_wtd_neigh_count);
    cudaFree(d_added_neigh_count);
    cudaFree(d_prod_neigh_count);

}
