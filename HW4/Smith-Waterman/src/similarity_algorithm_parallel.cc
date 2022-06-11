#include "similarity_algorithm_parallel.h"
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

using namespace Algorithms;

SimilarityAlgorithmParallel::SimilarityAlgorithmParallel(int seq1Length, int seq2Length): seq1Length(seq1Length), seq2Length(seq2Length)
{
}
/*--------------------------------------------------------------------
 * Function:    generate
 * Purpose:     Generate arrays a and b
 */


void SimilarityAlgorithmParallel::Run()
{
    FillMatrices();
    BackwardMoving();
}

void SimilarityAlgorithmParallel::PrintResults(std::string fileName)
{
  FILE* outf = fopen(fileName.c_str(), "w");

  fprintf(outf,"new: %d x %d\n", seq1Length+1, seq2Length+1);
  for(int i=0;i<(seq1Length+1);i++) { //print out matrix -> takes long time
    for(int j=0;j<(seq2Length+1);j++) {
      fprintf(outf,"%d\t",A[i][j]);
    }
    fprintf(outf,"\n");
  }

  fprintf(outf,"path len: %ld\n", path.size());
  int cnt=0;
  for(auto a: path) {
    fprintf(outf,"(%d,%d) -> ",a.first/*y*/, a.second/*x*/);
    if(++cnt %10==0) fprintf(outf,"\n");
  }
  fprintf(outf,"\n");
}

void SimilarityAlgorithmParallel::DeallocateMemoryForSingleRun()
{
    if(A != NULL)
    {
        if(A[0] != NULL)
            delete[] A[0];
        delete[] A;
    }
    if(E != NULL)
    {
        if(E[0] != NULL)
            delete[] E[0];
        delete[] E;
    }
    if(F != NULL)
    {
        if(F[0] != NULL)
            delete[] F[0];
        delete[] F;
    }
    if(B != NULL)
    {
        if(B[0] != NULL)
            delete[] B[0];
        delete[] B;
    }

    A = NULL;
    E = NULL;
    F = NULL;
    B = NULL;
}

SimilarityAlgorithmParallel::~SimilarityAlgorithmParallel()
{
    DeallocateMemoryForSingleRun();
}
