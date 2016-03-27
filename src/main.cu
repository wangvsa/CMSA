#include <stdio.h>
#include "util.h"
#include "center-star.h"
using namespace std;

__device__
void test(char *result, int num) {

}
__global__
void fun(char *result, int num) {
    test(result, num);
}

int main() {
    int vec[65536] = {0};

    list<string> sequences = readFastaFile("/home/wangchen/source/CUDA/CUDA-MSA/mt_genome_1x.fasta");
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

    printf("maxIndex: %d, maxCount:%d", maxIndex, maxCount);

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
