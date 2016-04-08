#include <fstream>
#include <assert.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
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
    printf("./msa [options] input_path output_path\n");
    printf("Options:\n");
    printf("\t-g\t: use GPU only (default use both GPU and CPU)\n");
    printf("\t-c\t: use CPU only (default use both GPU and CPU)\n");
    printf("\t-w <int>\t: specify the workload ratio of CPU / CPU\n");
    printf("\t-b <int>\t: specify the number of blocks per grid\n");
    printf("\t-t <int>\t: specify the number of threads per block\n");
    printf("\t-l <int>\t: specify the threshold to determine wheather use the register or global memory\n");
}


int parseOptions(int argc, char* argv[]) {
    if(argc < 3) {
        displayUsage();
        return -1;                          // 不执行程序
    }

    int oc;
    while((oc = getopt(argc, argv, "gcw:b:t:l:")) != -1) {
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
            case 'l':                       // 设置THRESHOLD
                THRESHOLD = atoi(optarg);
                break;
            case '?':                       // 输入错误选项，不执行程序
                displayUsage();
                return -1;
        }
    }

    return optind;
}
