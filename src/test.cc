#include <vector>
#include <string>
#include <stdio.h>
#include "util.h"
using namespace std;


int main() {
    vector<string> seqs = readFastaFile("/home/wangchen/source/CUDA/CUDA-MSA/16s_rRNA_small.fasta");
    vector<string> res;
    for(int i=0;i<seqs.size();i++) {
        string str = seqs[i].substr(0, 99);
        printf("length: %d\n", str.size());
        res.push_back(str);
    }
    writeFastaFile("/home/wangchen/test.fasta", res);
    return 0;
}
