#include <fstream>
#include <assert.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <cuda.h>
#include "util.h"
#include "global.h"
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

void displayUsage() {
    printf("Usage :\n");
    printf("./msa.out [options] input_path output_path\n");
    printf("Options:\n");
    printf("\t-g\t: use GPU only (default use both GPU and CPU)\n");
    printf("\t-c\t: use CPU only (default use both GPU and CPU)\n");
    printf("\t-w <int>\t: specify the workload ratio of CPU / CPU\n");
    printf("\t-b <int>\t: specify the number of blocks per grid\n");
    printf("\t-t <int>\t: specify the number of threads per block\n");
}


int parseOptions(int argc, char* argv[]) {
    if(argc < 3) {
        displayUsage();
        return -1;                          // 不执行程序
    }

    int oc;
    while((oc = getopt(argc, argv, "gcw:b:t:")) != -1) {
        switch(oc) {
            case 'g':                       // 只使用GPU
                MODE = GPU_ONLY;
                break;
            case 'c':                       // 只使用CPU (OpenMP)
                MODE = CPU_ONLY;
                break;
            case 'w':                       // 设置任务比例
                WORKLOAD_RATIO = atoi(optarg);
                break;
            case 'b':                       // 设置Blocks数量
                BLOCKS = atoi(optarg);
                break;
            case 't':                       // 设置Threads数量
                THREADS = atoi(optarg);
                break;
            case '?':                       // 输入错误选项，不执行程序
                displayUsage();
                return -1;
        }
    }

    return optind;
}


bool configureKernel(int centerSeqLength, int maxLength, unsigned long sumLength) {

    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);

    // 每次两两匹配的DP矩阵所需要的空间
    int matrixSize = sizeof(short) * (centerSeqLength+1) * (maxLength+1);

    // 得到每个Kernel可以执行的串数（即可并发的总线程数BLOCKS*THREADS）
    // 不应该使用所有的空闲内存，在此留出一部分（1000MB）
    freeMem = freeMem - sizeof(char) * sumLength - 1000*1024*1024;
    int seqs = freeMem / matrixSize;

    printf("freeMem: %dMB, sumLengthSize: %dMB, matrix :%dKB, seqs: %d\n", freeMem/1024/1024, sumLength/1024/1024, matrixSize/1024, seqs);

    // 先判断用户设置的<BLOCKS, THREADS>是否满足内存限制
    // 如果不满足，则自动设置一个<BLOCKS, THREADS>
    if(seqs >= BLOCKS*THREADS)
        return true;

    // 在满足内存限制的前提下，
    // 满足BLOCKS >= 3 且 THREADS >= 32则可以在GPU执行
    int b, t;
    for(t = THREADS; t >= 32; t -= 32) {
        b = seqs / t;
        if( b >= 3 && t >= 32) {
            BLOCKS = b;
            THREADS = t;
            return true;
        }
    }

    return false;
}
