#include "matmul.h"
#include <omp.h>

void matmul_ref(const int* const matrixA, const int* const matrixB,
                int* const matrixC, const int n, const int m) {
  // You can assume matrixC is initialized with zero
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < m; k++)
        matrixC[i * n + j] += matrixA[i * m + k] * matrixB[k * n + j];
}

void matmul_optimized(const int* const matrixA, const int* const matrixB,
                      int* const matrixC, const int n, const int m) {
  omp_set_num_threads(16);
  //transpose matrixB
  int* transMatrixB;
  transMatrixB = (int*)malloc(sizeof(int)*m*n);
  #pragma omp parallel for collapse(2)
  for (int i=0; i<n; i++) {
    for (int j=0; j<m; j++) {
      transMatrixB[i*m+j] = matrixB[j*n+i];
    }
  }
  // TODO: Implement your code

  int b = 32;
  if (n == 4096)
    b = 128;
  
  int checkLastA = 0;
  int checkLastB = 0;
  int checkLastK = 0;
  #pragma omp parallel for collapse(2)
  for (int i=0; i<n; i=i+b) {
    for (int j=0; j<n; j=j+b) {
      for (int k=0; k<m; k=k+8*b) {
        checkLastA = i+b;
        if (checkLastA > n)
          checkLastA = n;
        for (int ii=i; ii<checkLastA; ii++) {
          checkLastB = j+b;
          if (checkLastB > n)
            checkLastB = n;
          for (int jj=j; jj<checkLastB; jj++) {
            checkLastK = k+8*b;
            if (checkLastK > m)
              checkLastK = m;
            int accum = matrixC[ii*n+jj];
            for (int kk=k; kk<checkLastK; kk++) {
              accum += matrixA[ii*m+kk] * transMatrixB[jj*m+kk];
            }
            matrixC[ii*n+jj] = accum;
          }
        }
      }
    }
  }
}