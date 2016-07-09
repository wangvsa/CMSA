#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include "util.h"
#include "omp.h"
#include "cuda-nw.h"
#include "global.h"
using namespace std;

#define get_tid (threadIdx.x+blockIdx.x*blockDim.x)

/**
  * 打印矩阵
  * m 行, n 列
  */
__device__
void printMatrix(short *matrix, int m, int n) {
    for(int i=0;i<m;i++) {
        for(int j=0;j<n;j++)
            printf("%d ", matrix[i*m+j]);
        printf("\n");
    }
}

__device__
short max(short v1, short v2) {
    return v1 > v2 ? v1 : v2;
}
__device__
short max(short v1, short v2, short v3) {
    return max(max(v1, v2), v3);
}

__device__
int cuda_strlen(char *str) {
    int count = 0;
    while(str[count]!='\0')
        count++;
    return count;
}


/**
  * 每个线程有一个指向matrix的指针
  * matrix是一维的，大小是sizeof(short) * (m+1) * (maxLength+1)
  * 在堆中动态分配，每个kernel重复使用即可
  */
__device__ short* d_matrixPtr[MAX_THREADS * MAX_BLOCKS];

__global__
void allocDeviceMatrix(int centerSeqLength, int maxLength) {
    d_matrixPtr[get_tid] = new short[(centerSeqLength+1) * (maxLength+1)];
}

__global__
void freeDeviceMatrix() {
    if(d_matrixPtr[get_tid])
        delete[] d_matrixPtr[get_tid];
}



/**
  * m               in, 中心串长度
  * n               in, 对比串长度
  * centerSeq       in, 中心串
  * seqs            in, 其他n-1条串
  * seqIdx          in, 要被计算的串的编号
  * matrix          out, 需要计算的DP矩阵
  * 此函数没有被使用
  */
__device__
void cuda_nw(int m, int n, char *centerSeq, char *seq, short*matrix, int width) {
    // 初始化矩阵, DP矩阵m+1行,n+1列
    for(int i=0;i<=m;i++)
        matrix[i*width+0] = i * MISMATCH;   // matrix[i][0]
    for(int j=0;j<=n;j++)
        matrix[0*width+j] = j * MISMATCH;   // matrix[0][j];

    for(int i=1;i<=m;i++) {
        for(int j=1;j<=n;j++) {
            short up = matrix[(i-1)*width+j] + GAP;           // matrix[i-1][j]
            short left = matrix[i*width+j-1] + GAP;           // matrix[i][j-1]
            short diag = matrix[(i-1)*width+j-1] + ((centerSeq[i-1]==seq[j-1])?MATCH:MISMATCH);      // matrix[i-1][j-1]
            matrix[i*width+j] = max(up, left, diag);
        }
    }
}

#define COL_STEP 12
__device__
void cuda_nw_3d(int m, int n, char *centerSeq, char *seq, cudaPitchedPtr matrix3DPtr) {
    size_t slicePitch = matrix3DPtr.pitch * (m+1);
    char *slice = (char *)matrix3DPtr.ptr + get_tid * slicePitch;

    // 初始化矩阵, DP矩阵m+1行,n+1列
    DPCell *matrixRow;
    for(int i=0;i<=m;i++) {
        matrixRow = (DPCell *)(slice + i * matrix3DPtr.pitch);
        matrixRow[0].score = MIN_SCORE;   // matrix[i][0]
        matrixRow[0].x_gap = MIN_SCORE;
        matrixRow[0].y_gap = GAP_START + i * GAP_EXTEND;
    }
    matrixRow = (DPCell *)(slice + 0 * matrix3DPtr.pitch);
    for(int j=0;j<=n;j++) {
        matrixRow[j].score = MIN_SCORE;   // matrix[0][j];
        matrixRow[j].x_gap = GAP_START + j * GAP_EXTEND;
        matrixRow[j].y_gap = MIN_SCORE;
    }
    matrixRow[0].score = 0;             // matrix[0][0]


    /**
      * 参照这篇论文：
      * [IPDPS-2009]An Efficient Implementation Of Smith Waterman Algorithm On Gpu Using Cuda, For Massively Parallel Scanning Of Sequence Databases
      * 横向计算，每次计算COL_STEP列，理论上讲COL_STEP越大越好，取决与register per block的限制
      * 这样左侧依赖数据，以及一列（COL_STEP个cell）内的上侧依赖数据就可以存储在register中
      * 有效减少global memory访问次数。
      * TODO: 1. 对角线的global memory访问也可以节省掉
      *       2. 如果中心串的长度不能被COL_STEP整除怎么处理
      */
    short upScore, upYGap, diagScore;
    for(int i=1;i<=m;i+=COL_STEP) {
        // 直接这样生命没有把所有元素初始化为MIN_SCORE
        //short leftScore[COL_STEP] = {MIN_SCORE}, leftXGap[COL_STEP] = {MIN_SCORE};
        short leftScore[COL_STEP], leftXGap[COL_STEP];
        for(int tmp=0;tmp<COL_STEP;tmp++) {
            leftScore[tmp] = MIN_SCORE;
            leftXGap[tmp] = MIN_SCORE;
        }

        for(int j=1;j<=n;j++) {
            for(int k=0;k<COL_STEP;k++) {
                if(i+k>m) break;
                DPCell *matrixRow = (DPCell *)(slice + (i+k) * matrix3DPtr.pitch);
                DPCell *matrixLastRow = (DPCell *)(slice + (i-1+k) * matrix3DPtr.pitch);

                DPCell cell;            // 当前计算的cell
                if(k==0) {
                    upScore = matrixLastRow[j].score;
                    upYGap = matrixLastRow[j].y_gap;
                    diagScore = matrixLastRow[j-1].score;
                }

                cell.x_gap = max(GAP_START+GAP_EXTEND+leftScore[k], GAP_EXTEND+leftXGap[k]);
                cell.y_gap = max(GAP_START+GAP_EXTEND+upScore, GAP_EXTEND+upYGap);
                cell.score = diagScore + ((centerSeq[i+k-1]==seq[j-1])?MATCH:MISMATCH);               // matrix[i-1][j-1]
                cell.score = max(cell.x_gap, cell.y_gap, cell.score);

                // 更新当前列下一行cell计算所需要的数据
                upScore = cell.score;
                upYGap = cell.y_gap;
                diagScore = leftScore[k];
                // 更新当前行下一列cell计算所需要的数据
                leftScore[k] = cell.score;
                leftXGap[k] = cell.x_gap;

                matrixRow[j] = cell;    // 写入当前cell到Global Memory
            }
        }
    }
}

/**
  * m               in, 中心串长度, m 行
  * n               in, 对比串长度, n 列
  * seqIdx          in, 要被计算的串的编号
  * matrix          in, 本次匹配得到的DP矩阵
  * space           out, 需要计算的本次匹配给中心串引入的空格
  * spaceForOther   out, 需要计算的本次匹配给当前串引入的空格
  * 此函数没有被使用
  */
__device__
void cuda_backtrack(int m, int n, short* matrix, short *spaceRow, short *spaceForOtherRow, int width) {
    // 从(m, n) 遍历到 (0, 0)
    // DP矩阵的纬度是m+1, n+1
    int i = m, j = n;
    while(i!=0 || j!=0) {
        int score = matrix[i*width+j];                              // matrix[i][j]
        //printf("%d,%d:  %d\n", i, j, score);
        if(i > 0 && matrix[(i-1)*width+j] + GAP == score) {         // matrix[i-1][j]
            spaceForOtherRow[j]++;                                  // spaceForOther[seqIdx][j]
            i--;
        } else if(j > 0 && matrix[i*width+j-1] + GAP == score) {    // matrix[i][j-1]
            spaceRow[i]++;                                          // space[seqIdx][i]
            j--;
        } else {
            i--;
            j--;
        }
    }
}
__device__
void cuda_backtrack_3d(int m, int n, char *centerSeq, char *seq, cudaPitchedPtr matrix3DPtr, short *spaceRow, short *spaceForOtherRow) {
    size_t slicePitch = matrix3DPtr.pitch * (m+1);
    char *slice = (char *)matrix3DPtr.ptr + get_tid * slicePitch;

    int i = m, j = n;
    while(i!=0 || j!=0) {
        DPCell *matrixRow = (DPCell *)(slice + i * matrix3DPtr.pitch);
        DPCell *matrixLastRow = (DPCell *)(slice + (i-1) * matrix3DPtr.pitch);
        int score = (centerSeq[i-1] == seq[j-1]) ? MATCH : MISMATCH;
        if(i>0 && j>0 && score+matrixLastRow[j-1].score == matrixRow[j].score) {
            i--;
            j--;
        } else {
            int k = 1;
            while(true) {
                DPCell *matrixLastKRow = (DPCell *)(slice + (i-k) * matrix3DPtr.pitch);
                if(i>=k && matrixRow[j].score == matrixLastKRow[j].score+GAP_START+GAP_EXTEND*k) {
                    spaceForOtherRow[j] += k;
                    i = i - k;
                    break;
                } else if(j>=k && matrixRow[j].score == matrixRow[j-k].score+GAP_START+GAP_EXTEND*k) {
                    spaceRow[i] += k;
                    j = j - k;
                    break;
                } else {
                    k++;
                }
            }
        }
    }
}


__global__
void kernel(int startSeqIdx, char *centerSeq, char *seqs, int centerSeqLength, int *seqsSize, cudaPitchedPtr matrix3DPtr, short *space, short *spaceForOther, int maxLength, int workCount) {

    int tid = get_tid;
    int seqIdx = tid + startSeqIdx;
    if(seqIdx >= workCount) return;

    // 得到当前线程要计算的串
    int width = maxLength + 1;
    char *seq = seqs + width * seqIdx;

    //int m = cuda_strlen(centerSeq);
    //int n = cuda_strlen(seq);
    int m = centerSeqLength;
    int n = seqsSize[seqIdx];

    // 当前匹配的字符串所需要填的空格数组的位置
    short *spaceRow = space + tid * (m+1);
    short *spaceForOtherRow = spaceForOther + tid * width;

    // 计算使用的DP矩阵
    cuda_nw_3d(m, n, centerSeq, seq, matrix3DPtr);
    cuda_backtrack_3d(m, n,centerSeq, seq, matrix3DPtr, spaceRow, spaceForOtherRow);

    //printMatrix(spaceForOtherRow, 1, n+1);
}


/**
  * 支持多个GPU
  * workCount:int       需要由GPU执行的工作量，平均分给各个GPU
  * centerSeq:string    中心串
  * seqs:vector<string> 除中心串外的所有串
  * maxLength:int       所有串的最长长度
  */
void cuda_msa(int offset, int workCount, string centerSeq, vector<string> seqs, int maxLength, short *space, short *spaceForOther);
void multi_gpu_msa(int workCount, string centerSeq, vector<string> seqs, int maxLength, short *space, short *spaceForOther) {
    if(workCount<= 0) return;

    int GPU_NUM;
    cudaGetDeviceCount(&GPU_NUM);
    //GPU_NUM = 1;
    int workload = workCount / GPU_NUM;

    for(int i = 0; i < GPU_NUM; i++) {
        cudaSetDevice(i);
        if(i != GPU_NUM - 1) {
            cuda_msa(i*workload, workload, centerSeq, seqs, maxLength, space, spaceForOther);
        } else {                // 最后一块GPU还要做多做余数
            cuda_msa(i*workload, workload+(workCount%GPU_NUM), centerSeq, seqs, maxLength, space, spaceForOther);
        }
    }

    cudaDeviceReset();
}


void cuda_msa(int offset, int workCount, string centerSeq, vector<string> seqs, int maxLength, short *space, short *spaceForOther) {

    int sWidth = centerSeq.size() + 1;      // d_space的宽度
    int soWidth = maxLength + 1;            // d_spaceForOther的宽度

    // 1. 将中心串传到GPU
    char *d_centerSeq;
    cudaMalloc((void**)&d_centerSeq, sWidth * sizeof(char));
    cudaMemcpy(d_centerSeq, centerSeq.c_str(), sWidth *sizeof(char), cudaMemcpyHostToDevice);

    // 2. 将需要匹配串拼接成一个长串传到GPU
    char *d_seqs;
    cudaMalloc((void**)&d_seqs, (maxLength+1)*workCount*sizeof(char));
    char *c_seqs = new char[(maxLength+1) * workCount];
    for(int i=0;i<workCount;i++) {
        char *p = &(c_seqs[i * (maxLength + 1)]);
        strcpy(p, seqs[i+offset].c_str());
    }
    cudaMemcpy(d_seqs, c_seqs, (maxLength+1)*workCount*sizeof(char), cudaMemcpyHostToDevice);
    delete[] c_seqs;


    // 3. 将要匹配的串的长度也计算好传给GPU，因为在GPU上计算长度比较慢
    int *seqsSize = new int[workCount];
    for(int i = 0; i < workCount; i++)
        seqsSize[i] = seqs[i+offset].size();
    int *d_seqsSize;
    cudaMalloc((void**)&d_seqsSize, sizeof(int)*workCount);
    cudaMemcpy(d_seqsSize, seqsSize, sizeof(int)*workCount, cudaMemcpyHostToDevice);
    delete[] seqsSize;


    // 每个kernel计算SEQUENCES_PER_KERNEL条串
    int SEQUENCES_PER_KERNEL = BLOCKS * THREADS;
    int h = workCount < SEQUENCES_PER_KERNEL ? workCount : SEQUENCES_PER_KERNEL;

    // 给存储空格信息申请空间
    // d_space, d_spaceForOther 是循环利用的
    short *d_space, *d_spaceForOther;
    cudaMalloc((void**)&d_space, h*sWidth*sizeof(short));
    cudaMalloc((void**)&d_spaceForOther, h*soWidth*sizeof(short));


    // 分配一个3D的DP Matrix
    size_t freeMem, totalMem;
    cudaPitchedPtr matrix3DPtr;
    cudaExtent matrixSize = make_cudaExtent(sizeof(DPCell) * soWidth, sWidth, SEQUENCES_PER_KERNEL);
    cudaMalloc3D(&matrix3DPtr, matrixSize);
    cudaMemGetInfo(&freeMem, &totalMem);
    printf("freeMem :%luMB, totalMem: %luMB\n", freeMem/1024/1024, totalMem/1024/1024);

    for(int i = 0; i <= workCount / SEQUENCES_PER_KERNEL; i++) {
        if(i==workCount/SEQUENCES_PER_KERNEL)
            h = workCount % SEQUENCES_PER_KERNEL;

        // 此次kernel计算的起始串的位置（是相对位置，相对自己计算的起始串的）
        int startIdx = i * SEQUENCES_PER_KERNEL;
        printf("%d. idx: %d, h: %d\n", i, startIdx+offset, h);

        cudaMemset(d_space, 0, h*sWidth*sizeof(short));
        cudaMemset(d_spaceForOther, 0, h*soWidth*sizeof(short));

        kernel<<<BLOCKS, THREADS>>>(startIdx, d_centerSeq, d_seqs, centerSeq.size(), d_seqsSize, matrix3DPtr, d_space, d_spaceForOther, maxLength, workCount);
        cudaError_t err  = cudaGetLastError();
        if ( cudaSuccess != err )
            printf("Error: %d, %s\n", err, cudaGetErrorString(err));

        // 将空格信息传回给CPU
        // TODO：使用Pipeline可以重叠数据传输和kernel计算
        int spaceIdx = startIdx + offset;
        cudaMemcpy(space+spaceIdx*sWidth, d_space, h*sWidth*sizeof(short), cudaMemcpyDeviceToHost);
        cudaMemcpy(spaceForOther+spaceIdx*soWidth, d_spaceForOther, h*soWidth*sizeof(short), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_space);
    cudaFree(d_spaceForOther);
    cudaFree(d_centerSeq);
    cudaFree(d_seqs);
    cudaFree(matrix3DPtr.ptr);
}

