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
            int up = matrix[(i-1)*width+j] + GAP;           // matrix[i-1][j]
            int left = matrix[i*width+j-1] + GAP;           // matrix[i][j-1]
            int diag = matrix[(i-1)*width+j-1] + ((centerSeq[i-1]==seq[j-1])?MATCH:MISMATCH);      // matrix[i-1][j-1]
            matrix[i*width+j] = max(up, left, diag);
        }
    }

    //printMatrix(matrix, m, n);
}
__device__
void cuda_nw_3d(int m, int n, char *centerSeq, char *seq, cudaPitchedPtr matrix3DPtr) {
    size_t slicePitch = matrix3DPtr.pitch * (m+1);
    char *slice = (char *)matrix3DPtr.ptr + get_tid * slicePitch;

    // 初始化矩阵, DP矩阵m+1行,n+1列
    short *matrixRow;
    for(int i=0;i<=m;i++) {
        matrixRow = (short *)(slice + i * matrix3DPtr.pitch);
        matrixRow[0] = i * MISMATCH;   // matrix[i][0]
    }
    matrixRow = (short *)(slice + 0 * matrix3DPtr.pitch);
    for(int j=0;j<=n;j++) {
        matrixRow[j] = j * MISMATCH;   // matrix[0][j];
    }

    for(int i=1;i<=m;i++) {
        short *matrixLastRow = (short *)(slice + (i-1) * matrix3DPtr.pitch);
        short *matrixRow = (short *)(slice + i * matrix3DPtr.pitch);
        for(int j=1;j<=n;j++) {
            int up = matrixLastRow[j] + GAP;                                                // matrix[i-1][j]
            int left = matrixRow[j-1] + GAP;                                                // matrix[i][j-1]
            int diag = matrixLastRow[j-1] + ((centerSeq[i-1]==seq[j-1])?MATCH:MISMATCH);    // matrix[i-1][j-1]
            matrixRow[j] = max(up, left, diag);
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
void cuda_backtrack_3d(int m, int n, cudaPitchedPtr matrix3DPtr, short *spaceRow, short *spaceForOtherRow) {
    size_t slicePitch = matrix3DPtr.pitch * (m+1);
    char *slice = (char *)matrix3DPtr.ptr + get_tid * slicePitch;

    // 从(m, n) 遍历到 (0, 0)
    // DP矩阵的纬度是m+1, n+1
    int i = m, j = n;
    while(i!=0 || j!=0) {
        short *matrixLastRow = (short *)(slice + (i-1) * matrix3DPtr.pitch);
        short *matrixRow = (short *)(slice + i * matrix3DPtr.pitch);
        int score = matrixRow[j];                                   // matrix[i][j]
        //printf("%d,%d:  %d\n", i, j, score);
        if(i > 0 && matrixLastRow[j] + GAP == score) {              // matrix[i-1][j]
            spaceForOtherRow[j]++;                                  // spaceForOther[seqIdx][j]
            i--;
        } else if(j > 0 && matrixRow[j-1] + GAP == score) {         // matrix[i][j-1]
            spaceRow[i]++;                                          // space[seqIdx][i]
            j--;
        } else {
            i--;
            j--;
        }
    }
}


__global__
void kernel(int startIdx, char *centerSeq, char *seqs, int centerSeqLength, int *seqsSize, cudaPitchedPtr matrix3DPtr, short *space, short *spaceForOther, int maxLength, int totalSequences) {

    int tid = get_tid;
    int seqIdx = tid + startIdx;
    if(seqIdx >= totalSequences) return;

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
    if(!matrix3DPtr.ptr) {
        short* matrix = d_matrixPtr[tid];
        cuda_nw(m, n, centerSeq, seq, matrix, width);
        cuda_backtrack(m, n, matrix, spaceRow, spaceForOtherRow, width);
    } else {
        cuda_nw_3d(m, n, centerSeq, seq, matrix3DPtr);
        cuda_backtrack_3d(m, n, matrix3DPtr, spaceRow, spaceForOtherRow);
    }

    //printMatrix(spaceForOtherRow, 1, n+1);
}


void cuda_msa(int workCount, string centerSeq, vector<string> seqs, int maxLength, short *space, short *spaceForOther) {
    if(workCount<= 0) return;

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
    cudaMalloc((void**)&d_seqs, soWidth*workCount*sizeof(char));
    char *c_seqs = new char[(maxLength+1) * workCount];
    for(int i=0;i<workCount;i++) {
        char *p = &(c_seqs[i * (maxLength + 1)]);
        strcpy(p, seqs[i].c_str());
    }
    cudaMemcpy(d_seqs, c_seqs, soWidth*workCount*sizeof(char), cudaMemcpyHostToDevice);
    delete[] c_seqs;


    int *seqsSize = new int[seqs.size()];
    for(int i = 0; i < seqs.size(); i++)
        seqsSize[i] = seqs[i].size();
    int *d_seqsSize;
    cudaMalloc((void**)&d_seqsSize, sizeof(int)*seqs.size());
    cudaMemcpy(d_seqsSize, seqsSize, sizeof(int)*seqs.size(), cudaMemcpyHostToDevice);
    delete[] seqsSize;


    // d_space, d_spaceForOther 是循环利用的
    // 每个kernel计算SEQUENCES_PER_KERNEL条串
    int SEQUENCES_PER_KERNEL = BLOCKS * THREADS;
    int h = workCount < SEQUENCES_PER_KERNEL ? workCount : SEQUENCES_PER_KERNEL;

    // 每条串一个空格数组
    short *d_space;
    short *d_spaceForOther;
    cudaMalloc((void**)&d_space, h*sWidth*sizeof(short));
    cudaMalloc((void**)&d_spaceForOther, h*soWidth*sizeof(short));


    // 在堆中动态分配matrix所需要的内存（有4GB的上限）
    // 或者直接分配一个3D的Matrix
    size_t freeMem, totalMem;
    cudaPitchedPtr matrix3DPtr;
    if(USE_HEAP) {
        matrix3DPtr.ptr = NULL;
        // 设置(可用内存的80%)堆内存的上限，大于4GB会出错
        cudaMemGetInfo(&freeMem, &totalMem);
        size_t heapSize = freeMem/10*8 > 4UL*1024*1024*1024 ? 4UL*1024*1024*1024 : freeMem/10*8;
        cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize);
        allocDeviceMatrix<<<BLOCKS, THREADS>>>(centerSeq.size(), maxLength);
    } else {
        cudaExtent matrixSize = make_cudaExtent(sizeof(short) * soWidth, sWidth, SEQUENCES_PER_KERNEL);
        cudaMalloc3D(&matrix3DPtr, matrixSize);
    }
    cudaMemGetInfo(&freeMem, &totalMem);
    printf("freeMem :%luMB, totalMem: %luMB\n", freeMem/1024/1024, totalMem/1024/1024);



    for(int i = 0; i <= workCount / SEQUENCES_PER_KERNEL; i++) {
        if(i==workCount/SEQUENCES_PER_KERNEL)
            h = workCount % SEQUENCES_PER_KERNEL;

        // 此次kernel计算的起始串的位置
        int startIdx = i * SEQUENCES_PER_KERNEL;
        printf("%d. startIdx: %d, h: %d\n", i, startIdx, h);

        cudaMemset(d_space, 0, h*sWidth*sizeof(short));
        cudaMemset(d_spaceForOther, 0, h*soWidth*sizeof(short));

        kernel<<<BLOCKS, THREADS>>>(startIdx, d_centerSeq, d_seqs, centerSeq.size(), d_seqsSize, matrix3DPtr, d_space, d_spaceForOther, maxLength, workCount);
        cudaError_t err  = cudaGetLastError();
        if ( cudaSuccess != err )
            printf("Error: %d, %s\n", err, cudaGetErrorString(err));

        cudaMemcpy(space+startIdx*sWidth, d_space, h*sWidth*sizeof(short), cudaMemcpyDeviceToHost);
        cudaMemcpy(spaceForOther+startIdx*soWidth, d_spaceForOther, h*soWidth*sizeof(short), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_space);
    cudaFree(d_spaceForOther);
    cudaFree(d_centerSeq);
    cudaFree(d_seqs);
    if(USE_HEAP)
        freeDeviceMatrix<<<BLOCKS, THREADS>>>();
    else
        cudaFree(matrix3DPtr.ptr);
}
