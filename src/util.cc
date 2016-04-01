#include "util.h"
#include <fstream>
#include <assert.h>
using namespace std;

vector<string> readFastaFile(const char *path) {
    vector<string> sequences;
    string buff;

    ifstream file;
    file.open(path);
    assert(file);

    while(getline(file, buff)) {
        if(buff[0] == '>')
            continue;
        sequences.push_back(buff);
    }

    file.close();
    return sequences;
}


void writeFastaFile(const char* path, vector<string> strs) {
    ofstream file(path);
    if(file.is_open()) {
        for(int i=0;i<strs.size();i++) {
            file<<">"<<i<<endl;
            file<<strs[i]<<endl;
        }
    }

    file.close();
}
