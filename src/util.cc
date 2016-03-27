#include "util.h"
#include <fstream>
#include <assert.h>
using namespace std;

list<string> readFastaFile(const char *path) {
    list<string> sequences;
    string buff;

    ifstream file;
    file.open(path);
    assert(file);

    while(getline(file, buff)) {
        if(buff[0] == '>')
            continue;
        sequences.push_back(buff);
    }

    return sequences;
}

