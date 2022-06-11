#ifndef _SMITH_WATERMAN_PARALLEL_H_
#define _SMITH_WATERMAN_PARALLEL_H_

#include <climits>
#include "similarity_algorithm_parallel.h"

namespace Algorithms
{
        class SmithWatermanParallel:public SimilarityAlgorithmParallel
  {
    public:
    SmithWatermanParallel(int seq1Length, int seq2Length, char* seq1, char*seq2, int gapOp, int gapEx);
    protected:
    virtual void setSeq1(char* seq, int len) {this->seq1 = seq; this->seq1Length = len;}
    virtual void setSeq2(char* seq, int len) {this->seq2 = seq; this->seq2Length = len;}
    virtual void FillMatrices();
    virtual void FillCell(int i, int j);
    virtual void RecursiveFill(int i, int j, int blockHeight, int blockWidth);
    virtual void BackwardMoving();
    virtual int matchMissmatchScore(char, char);
    int maxX;
    int maxY;
    int maxVal;
    const int matchScore = 5;
    const int missmatchScore = -3;
    int gapOp;  // gap open
    int gapEx;  // gap extension
  };
}

#endif
