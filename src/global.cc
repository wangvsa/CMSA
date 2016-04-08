#include "global.h"

int BLOCKS = 6;
int THREADS = 64;
int THRESHOLD = 100;

int WORKLOAD_RATIO = 1; // 默认GPU/CPU任务比例1:1，即各自负责一半的串

int MODE = CPU_GPU;     // 默认同时使用GPU和CPU
