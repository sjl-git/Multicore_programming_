#include "smith_waterman_parallel.h"
//#include <algorithm>
#include "utils.h"
#include <stdio.h>
#include <omp.h>
#include <thread>
#include <iostream>

using namespace Algorithms;

SmithWatermanParallel::SmithWatermanParallel(int seq1Length, int seq2Length, char* seq1, char* seq2, int gapOp, int gapEx):SimilarityAlgorithmParallel(seq1Length, seq2Length), gapOp(gapOp), gapEx(gapEx)
{
  A = new int*[seq1Length + 1];
  E = new int*[seq1Length + 1]; //left matrix
  F = new int*[seq1Length + 1]; //up matrix
  B = new BackUpStruct*[seq1Length + 1];
  A[0] = new int[(seq1Length + 1) * (seq2Length + 1)](); //intializ to 0
  E[0] = new int[(seq1Length + 1) * (seq2Length + 1)]();
  F[0] = new int[(seq1Length + 1) * (seq2Length + 1)]();
  B[0] = new BackUpStruct[(seq1Length + 1) * (seq2Length + 1)]();

  for (int i = 1; i < seq1Length + 1; i++)
  {
    A[i] = A[0] + (seq2Length + 1)*i;
    E[i] = E[0] + (seq2Length + 1)*i;
    F[i] = F[0] + (seq2Length + 1)*i;
    B[i] = B[0] + (seq2Length + 1)*i;
  }

  setSeq1(seq1, seq1Length);
  setSeq2(seq2, seq2Length);
}

int SmithWatermanParallel::matchMissmatchScore(char a, char b) {
  if (a == b)
    return matchScore;
  else
    return missmatchScore;
}  /* End of matchMissmatchScore */


void SmithWatermanParallel::FillCell(int i, int j)
{
      E[i][j] = MAX(E[i][j - 1] - gapEx, A[i][j - 1] - gapOp);
      B[i][j - 1].continueLeft = (E[i][j] == E[i][j - 1] - gapEx);
      F[i][j] = MAX(F[i - 1][j] - gapEx, A[i - 1][j] - gapOp);
      B[i - 1][j].continueUp = (F[i][j] == F[i - 1][j] - gapEx);

      A[i][j] = MAX3(E[i][j], F[i][j], A[i - 1][j - 1] + matchMissmatchScore(seq1[i-1], seq2[j-1]));
      A[i][j] = MAX(A[i][j], 0);

      if (A[i][j] == 0)
        B[i][j].backDirection = stop; //SPECYFIC FOR SMITH WATERMAN
      else if(A[i][j] == (A[i - 1][j - 1] + matchMissmatchScore(seq1[i-1], seq2[j-1])))
        B[i][j].backDirection = crosswise;
      else if(A[i][j] == E[i][j])
        B[i][j].backDirection = left;
      else //if(A[i][j] == F[i][j])
        B[i][j].backDirection = up;

      if(A[i][j] > maxVal)
      {
        maxX = j;
        maxY = i;
        maxVal = A[i][j];
      }
}

void SmithWatermanParallel::RecursiveFill(int i, int j, int blockHeight, int blockWidth)
{
  for (int k = i; k < i+blockHeight; k++)
  {
    for (int l = j; l < j+blockWidth; l++)
    {
      FillCell(k, l);
    }
    
  }

  if (i+blockHeight <= seq1Length) {
    int checkIdx = i + 2*blockHeight - 1;
    int newBlockHeight = blockHeight;
    if (seq1Length - i + 1 < 2*blockHeight) {
      checkIdx = seq1Length;
      newBlockHeight = seq1Length - i - blockHeight + 1;
    }
    if (j == 1 || (B[checkIdx][j-1].backDirection != unfilled && B[checkIdx][j-1].backDirection != temporal) ) {
      if (B[i+blockHeight][j].backDirection == unfilled) {
        B[checkIdx][j+blockWidth-1].backDirection = temporal;
        #pragma omp task
        RecursiveFill(i+blockHeight, j, newBlockHeight, blockWidth);
      }
    }
  }
  if (j+blockWidth <= seq2Length) {
    int checkIdx = j+2*blockWidth-1;
    int newBlockWidth = blockWidth;
    if (seq2Length - j + 1 < 2*blockWidth) {
      checkIdx = seq2Length;
      newBlockWidth = seq2Length - j - blockWidth + 1;
    }
    if (i == 1 || (B[i-1][checkIdx].backDirection != unfilled && B[i-1][checkIdx].backDirection != temporal)) {
      if (B[i][j+blockWidth].backDirection == unfilled) {
        B[i+blockHeight-1][checkIdx].backDirection = temporal;
        #pragma omp task
        RecursiveFill(i, j+blockWidth, blockHeight, newBlockWidth);
      }
    }
  }
}


void SmithWatermanParallel::FillMatrices()
{
  /*
   *   s e q 2
   * s
   * e
   * q
   * 1
   */
  //E - responsible for left direction
  //F - responsible for up   direction
  maxVal = INT_MIN;

  int blockHeight = 1;
  int blockWidth = 1;
   
  if (seq1Length >= 5000) {
    blockHeight = 1000;
  } else if (seq1Length >= 500) {
    blockHeight = 100;
  } else if (seq1Length >= 50) {
    blockHeight = 10;
  }
  if (seq2Length >= 5000) {
    blockWidth = 1000;
  } else if (seq2Length >= 500) {
    blockWidth = 100;
  } else if (seq2Length >= 50) {
    blockWidth = 10;
  }

  omp_set_num_threads(16);
  #pragma omp parallel
  #pragma omp single
  RecursiveFill(1, 1, blockHeight, blockWidth);
  #pragma omp taskwait
  
  printf("Matrix Filled: maxY %d maxX %d maxVal %d\n", maxY, maxX, maxVal);
}

void SmithWatermanParallel::BackwardMoving()
{
  //BACKWARD MOVING
  int carret = 0;

  int y = maxY;
  int x = maxX;

  BackDirection prev = crosswise;
  while(B[y][x].backDirection != stop && B[y][x].backDirection != unfilled && B[y][x].backDirection != temporal)
  {
    path.push_back(std::make_pair(y, x));
    if (prev == up && B[y][x].continueUp) //CONTINUE GOING UP
    {                                          //GAP EXTENSION
      carret++;
      y--;
    }
    else if (prev == left && B[y][x].continueLeft) //CONTINUE GOING LEFT
    {                                         //GAP EXTENSION
      carret++;
      x--;
    }
    else
    {
      prev = B[y][x].backDirection;
      if(prev == up)
      {
        carret++;
        y--;
      }
      else if(prev == left)
      {
        carret++;
        x--;
      }
      else //prev == crosswise
      {
        carret++;
        x--;
        y--;
      }
    }
  }
  printf("Backward Moving: destY %d destX %d\n", y, x);
}
