#ifndef _GLOBAL_H_
#define _GLOBAL_H_

/**
 * 定义全局变量
 */

#define MISMATCH -1
#define MATCH 0
#define GAP -1

#define MAX_THREADS 1024
#define MAX_BLOCKS 128


// 每个Kernel中的Block数量
extern int BLOCKS;

// 每个Block中Thread数量
extern int THREADS;

// 任务分配的比例, GPU/CPU
extern int WORKLOAD_RATIO;

// 运行方式
#define GPU_ONLY 1      // 只使用GPU
#define CPU_ONLY 2      // 只使用CPU
#define CPU_GPU 3       // 同时使用GPU和CPU
extern int MODE;        // 1, 2, 3

#endif
