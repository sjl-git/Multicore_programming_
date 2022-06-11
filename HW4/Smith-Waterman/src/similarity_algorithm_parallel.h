#ifndef _SIMILARITY_ALGORITHM_CPU_PARALLEL_H_
#define _SIMILARITY_ALGORITHM_CPU_PARALLEL_H_

#include "back_up_struct.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>
#include <string>
#include <utility>


namespace Algorithms
{
        class SimilarityAlgorithmParallel
        {
        public:
            virtual void Run();
            virtual void PrintResults(std::string fileName);
	    virtual void DeallocateMemoryForSingleRun();
            SimilarityAlgorithmParallel(int seq1Length, int seq2Length);
            ~SimilarityAlgorithmParallel();

        protected:
            virtual void FillMatrices() = 0;
            virtual void BackwardMoving() = 0;

            //INPUT DATA
            // seq1 and seq2 are 1-based array of sequence characters
            char* seq1;
            char* seq2;
            int seq1Length;
            int seq2Length;

            //MATRICES
            int **A;
            int **E; //left matrix
            int **F; //up matrix
            BackUpStruct **B;

            //backtrack results
            std::vector<std::pair<int,int> > path;
        };
}

#endif
