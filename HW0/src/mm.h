#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <vector>

using namespace std;

class MatchMaker {
public:
    MatchMaker(string input_path);
    ~MatchMaker();

    void Match();
    void CheckAnswer(string answer_path);
private:
    char* ref_str;
    char** query;
    int* output;

    int ref_len;
    int* query_len;
    int output_len;
    
    void MakeOutputFile();
};