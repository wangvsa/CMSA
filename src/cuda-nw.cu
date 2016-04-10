#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include "util.h"
#include "omp.h"
#include "cuda-nw.h"
#include "global.h"
using namespace std;


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
int max(int v1, int v2, int v3) {
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
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    d_matrixPtr[tid] = (short*)malloc(sizeof(short) * (centerSeqLength+1) * (maxLength+1));
}

__global__
void freeDeviceMatrix() {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(d_matrixPtr[tid])
        free(d_matrixPtr[tid]);
}



/**
  * centerSeq       in, 中心串
  * seqs            in, 其他n-1条串
  * seqIdx          in, 要被计算的串的编号
  * matrix          out, 需要计算的DP矩阵
  */
__device__
void cuda_nw(int m, int n, char *centerSeq, char *seq, int width) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    short *matrix = d_matrixPtr[tid];

    // 初始化矩阵, DP矩阵m+1行,n+1列
    for(int i=0;i<=m;i++)
        matrix[i*width+0] = i * MISMATCH;   // matrix[i][0]
    for(int j=0;j<=n;j++)
        matrix[0*width+j] = j * MISMATCH;   // matrix[0][j];

    for(int i=1;i<=m;i++) {
        for(int j=1;j<=n;j++) {
            int up = matrix[(i-1)*width+j] + GAP;           // matrix[i-1][j]
            int left = matrix[i*width+j-1] + GAP;           // matrix[i][j-1]
            int diag = matrix[(i-1)*width+j-1] + ((centerSeq[i-1]==seq[j-1])?MATCH:MISMATCH);      // matrix[i-1][j-1]
            matrix[i*width+j] = max(up, left, diag);
        }
    }

    //printMatrix(matrix, m, n);
}

/**
  * m               in, 中心串长度, m 行
  * n               in, 对比串长度, n 列
  * seqIdx          in, 要被计算的串的编号
  * matrix          in, 本次匹配得到的DP矩阵
  * space           out, 需要计算的本次匹配给中心串引入的空格
  * spaceForOther   out, 需要计算的本次匹配给当前串引入的空格
  */
__device__
void cuda_backtrack(int m, int n, int seqIdx, short *spaceRow, short *spaceForOtherRow, int width) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    short *matrix = d_matrixPtr[tid];

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


__global__
void kernel(int startIdx, char *centerSeq, char *seqs, short *space, short *spaceForOther, int maxLength, int totalSequences, int THRESHOLD) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int seqIdx = tid + startIdx;
    if(seqIdx >= totalSequences) return;

    // 得到当前线程要计算的串
    int width = maxLength + 1;
    char *seq = seqs + width * seqIdx;

    int m = cuda_strlen(centerSeq);
    int n = cuda_strlen(seq);

    // 当前匹配的字符串所需要填的空格数组的位置
    short *spaceRow = space + tid * (m+1);
    short *spaceForOtherRow = spaceForOther + tid * width;

    cuda_nw(m, n, centerSeq, seq, width);
    cuda_backtrack(m, n, tid, spaceRow, spaceForOtherRow, width);

    //printMatrix(spaceForOtherRow, 1, n+1);
}


void cuda_msa(int BLOCKS, int THREADS, int maxLength, int height, string centerSeq, vector<string> seqs, short *space, short *spaceForOther) {
    if(height <= 0) return;

    int sWidth = centerSeq.size() + 1;      // d_space的宽度
    int soWidth = maxLength + 1;            // d_spaceForOther的宽度
    // 共有seqs.size()条串，在GPU上计算其中的height条串
    // 剩余的交由CPU上使用openmp计算
    //int height = seqs.size();

    // 给字符串分配空间
    char *d_centerSeq;
    cudaMalloc((void**)&d_centerSeq, sWidth * sizeof(char));
    cudaMemcpy(d_centerSeq, centerSeq.c_str(), sWidth *sizeof(char), cudaMemcpyHostToDevice);
    char *d_seqs;
    cudaMalloc((void**)&d_seqs, soWidth*height*sizeof(char));
    char *c_seqs = new char[(maxLength+1) * height];
    for(int i=0;i<height;i++) {
        char *p = &(c_seqs[i * (maxLength + 1)]);
        strcpy(p, seqs[i].c_str());
    }
    cudaMemcpy(d_seqs, c_seqs, soWidth*height*sizeof(char), cudaMemcpyHostToDevice);
    delete[] c_seqs;

    // d_space, d_spaceForOther 是循环利用的
    // 每个kernel计算SEQUENCES_PER_KERNEL条串
    int SEQUENCES_PER_KERNEL = BLOCKS * THREADS;
    int h = height < SEQUENCES_PER_KERNEL ? height : SEQUENCES_PER_KERNEL;

    // 每条串一个空格数组
    short *d_space;
    short *d_spaceForOther;
    cudaMalloc((void**)&d_space, h*sWidth*sizeof(short));
    cudaMalloc((void**)&d_spaceForOther, h*soWidth*sizeof(short));


    // 设置(可用内存-500M)堆内存的上限
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, freeMem-(500*1024*1024));
    printf("freeMem :%dMB, totalMem: %dMB\n", freeMem/1024/1024, totalMem/1024/1024);

    // 在堆中分配matrix所需要的内存
    allocDeviceMatrix<<<BLOCKS, THREADS>>>(centerSeq.size(), maxLength);
    cudaMemGetInfo(&freeMem, &totalMem);
    printf("freeMem :%dMB, totalMem: %dMB\n", freeMem/1024/1024, totalMem/1024/1024);

    for(int i = 0; i <= height / SEQUENCES_PER_KERNEL; i++) {
        if(i==height/SEQUENCES_PER_KERNEL)
            h = height % SEQUENCES_PER_KERNEL;

        // 此次kernel计算的起始串的位置
        int startIdx = i * SEQUENCES_PER_KERNEL;
        printf("%d. startIdx: %d, h: %d\n", i, startIdx, h);

        cudaMemset(d_space, 0, h*sWidth*sizeof(short));
        cudaMemset(d_spaceForOther, 0, h*soWidth*sizeof(short));

        kernel<<<BLOCKS, THREADS>>>(startIdx, d_centerSeq, d_seqs, d_space, d_spaceForOther, maxLength, height, THRESHOLD);
        cudaError_t err  = cudaGetLastError();
        if ( cudaSuccess != err )
            printf("Error: %d, %s\n", err, cudaGetErrorString(err));

        cudaMemcpy(space+startIdx*sWidth, d_space, h*sWidth*sizeof(short), cudaMemcpyDeviceToHost);
        cudaMemcpy(spaceForOther+startIdx*soWidth, d_spaceForOther, h*soWidth*sizeof(short), cudaMemcpyDeviceToHost);
    }

    freeDeviceMatrix<<<BLOCKS, THREADS>>>();
    cudaFree(d_space);
    cudaFree(d_spaceForOther);
    cudaFree(d_centerSeq);
    cudaFree(d_seqs);
}