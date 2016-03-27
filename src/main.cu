#include <stdio.h>
#include "util.h"
#include "center-star.h"
#include "sp.h"
using namespace std;

__device__
void test(char *result, int num) {

}
__global__
void fun(char *result, int num) {
    test(result, num);
}

// 计算Sum of Pairs
void calSP(list<string> seqs) {
    int sp = sumOfPairs(seqs);
    printf("sp:%d, avg sp: %d\n", sp, sp/seqs.size());
}

// 找到中心串
void findCenterSequence(list<string> sequences) {
    int vec[65536] = {0};

    list<string>::iterator it;
    for(it=sequences.begin();it!=sequences.end();it++) {
        const char *str = (*it).c_str();
        setOccVector(str, vec);
    }

    int i = 1;

    int maxIndex = 0, maxCount = 0;
    for(it=sequences.begin();it!=sequences.end();it++) {
        const char *str = (*it).c_str();
        int count = countSequences(str, vec);
        if(count > maxCount) {
            maxIndex = i;
            maxCount = count;
        }
        printf("seq: %d, count: %d,\n", i++, count);
    }

    printf("maxIndex: %d, maxCount:%d\n", maxIndex, maxCount);
}

int main() {

    list<string> sequences = readFastaFile("/home/wangchen/source/CUDA/CUDA-MSA/test.fasta");

    findCenterSequence(sequences);

    calSP(sequences);

    /*
    char *d_result;
    cudaMalloc(&d_result, 10*sizeof(char));

    fun<<<2, 512>>>(d_result, 10);

    char *result = (char *)malloc(sizeof(char) * 11);
    memset(result, '\0', sizeof(char)*11);
    cudaMemcpy(result, d_result, sizeof(char)*10, cudaMemcpyDeviceToHost);
    printf("result:%s\n", result);

    free(result);
    cudaFree(d_result);
    */
}


