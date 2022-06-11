#ifndef MATMUL_H
#define MATMUL_H

void matmul_ref(const int* const matrixA, const int* const matrixB,
                int* const matrixC, const int n);

void matmul_optimized(const int* const matrixA, const int* const matrixB,
                      int* const matrixC, const int* const d_A, const int* const d_B,  int* const d_C, const int n) ;

void allocateDeviceMemory(void** M, int width);
void deallocateDeviceMemory(void* M);
#endif
