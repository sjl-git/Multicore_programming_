#include "mm.h"

using namespace std;

int main(int argc, char** argv) {
    MatchMaker mm(argv[1]);

    mm.Match();
    mm.CheckAnswer(argv[2]);
    return 0;
}