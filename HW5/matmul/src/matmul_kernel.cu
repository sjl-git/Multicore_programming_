#include <stdio.h>
#include <iostream>
#include <chrono>
#include <assert.h>
#include "matmul.h"
using namespace std;

#define TILE_WIDTH 32

void allocateDeviceMemory(void** M, int size)
{
  cudaError_t err = cudaMalloc(M, size);
  assert(err==cudaSuccess);
}

void deallocateDeviceMemory(void* M)
{
  cudaError_t err = cudaFree(M);
  assert(err==cudaSuccess);
}

void matmul_ref(const int* const matrixA, const int* const matrixB,
                int* const matrixC, const int n) {
  // You can assume matrixC is initialized with zero
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < n; k++)
        matrixC[i * n + j] += matrixA[i * n + k] * matrixB[k * n + j];
}

__global__ void matrix_mul_kernel(const int* const M, const int* const N, int* const P, int width) {

  __shared__ int subTileM[TILE_WIDTH][TILE_WIDTH];
  __shared__ int subTileN[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int Row = by*TILE_WIDTH + ty;
  int Col = bx*TILE_WIDTH + tx;
  int Pvalue = 0;
  int gridSize = ceil((float)width/(float)TILE_WIDTH);
  for (int m=0; m < gridSize; ++m) {
      subTileM[ty][tx] = M[Row*width + m*TILE_WIDTH+tx];
      subTileN[ty][tx] = N[(m*TILE_WIDTH+ty)*width+Col];
      __syncthreads();
    if (Row < width && Col < width) {
      for (int k=0; k < TILE_WIDTH; ++k) {
        Pvalue += subTileM[ty][k] * subTileN[k][tx];
      }
    }
      __syncthreads();
  }
  if (Row < width && Col < width) {
    P[Row * width + Col] = Pvalue;
  }
}

void matmul_optimized(const int* const matrixA, const int* const matrixB,
                      int* const matrixC, const int* const d_A, const int* const d_B,  int* const d_C, const int n) {

  // TODO: Implement your CUDA code
  int size = n*n*sizeof(int);

  cudaMemcpy((void *)d_A, (void *)matrixA, size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_B, (void *)matrixB, size, cudaMemcpyHostToDevice);
  int gridSize = ceil((float)n/(float)TILE_WIDTH);
  dim3 dimGrid(gridSize, gridSize, 1);
  dim3 dimBlock(TILE_WIDTH,TILE_WIDTH,1);

  matrix_mul_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);
  cudaMemcpy(matrixC, d_C, size, cudaMemcpyDeviceToHost);
}