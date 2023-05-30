/*
#include <iostream>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>

using namespace std;

#define BLOCK_SIZE 256

__global__ void vectorAddKernel(int *a, int *b, int *c, int size)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < size)
    {
        c[index] = a[index] + b[index];
    }
}

void vectorAdd(int *a, int *b, int *c, int size)
{
    int *d_a, *d_b, *d_c;
    int memSize = size * sizeof(int);
    cudaMalloc(&d_a, memSize);
    cudaMalloc(&d_b, memSize);
    cudaMalloc(&d_c, memSize);

    cudaMemcpy(d_a, a, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, memSize, cudaMemcpyHostToDevice);

    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, size);

    cudaMemcpy(c, d_c, memSize, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main()
{
    int size = 1000000;
    int *a = new int[size];
    int *b = new int[size];
    int *c = new int[size];

    // Input arrays initialization
    srand(time(NULL));
    for (int i = 0; i < size; i++)
    {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }

    // CUDA vector addition
    clock_t start = clock();
    vectorAdd(a, b, c, size);
    clock_t end = clock();

    // Output of the result vector
    cout << "Result vector: ";
    for (int i = 0; i < size; i++)
    {
        cout << c[i] << " ";
    }
    cout << endl;

    // Time taken by CUDA vector addition
    cout << "Time taken: " << double(end - start) / CLOCKS_PER_SEC << " seconds." << endl;

    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}
*/





#include <iostream>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>

using namespace std;

#define BLOCK_SIZE 16

__global__ void matrixMulKernel(int *a, int *b, int *c, int size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < size && col < size)
    {
        int sum = 0;
        for (int i = 0; i < size; i++)
        {
            sum += a[row * size + i] * b[i * size + col];
        }
        c[row * size + col] = sum;
    }
}

void matrixMul(int *a, int *b, int *c, int size)
{
    int *d_a, *d_b, *d_c;
    int memSize = size * size * sizeof(int);
    cudaMalloc(&d_a, memSize);
    cudaMalloc(&d_b, memSize);
    cudaMalloc(&d_c, memSize);

    cudaMemcpy(d_a, a, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, memSize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((size + BLOCK_SIZE - 1) / BLOCK_SIZE, (size + BLOCK_SIZE - 1) / BLOCK_SIZE
}






/*
g++ -fopenmp HPC_Assg_04.cpp -o HPC_Assg_04
./HPC_Assg_04




g++ -fopenmp -I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\include" HPC_Assg_04.cpp -o HPC_Assg_04


g++ -fopenmp -I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\include" HPC_Assg_04.cpp -o HPC_Assg_04 -L "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\lib\x64" -lcudart




Vector Addition:
Input:

a = [1, 2, 3, 4, 5]
b = [5, 4, 3, 2, 1]
n = 5

Output:
c = [6, 6, 6, 6, 6]


Matrix Multiplication:
Input:

A = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]

B = [[9, 8, 7],
     [6, 5, 4],
     [3, 2, 1]]


Output:

C = [[30, 24, 18],
     [84, 69, 54],
     [138, 114, 90]]

*/

