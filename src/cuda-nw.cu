#include <stdio.h>
#include <string>
#include <vector>
#include "util.h"
#include "omp.h"
using namespace std;

#define MISMATCH -1
#define MATCH 0
#define GAP -1


int maxLength;
string centerSeq;
vector<string> seqs;
char *c_seqs;


/**
  * 打印矩阵
  * m 行, n 列
  */
__device__
void printMatrix(int *matrix, int m, int n) {
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
  * centerSeq       in, 中心串
  * seqs            in, 其他n-1条串
  * seqIdx          in, 要被计算的串的编号
  * matrix          out, 需要计算的DP矩阵
  */
__device__
void cuda_nw(char *centerSeq, char *seq, int *matrix, int maxLength) {
    // m行, n列
    int m = cuda_strlen(centerSeq) + 1;
    int n = cuda_strlen(seq) + 1;
    int width = maxLength + 1;

    // 初始化矩阵
    for(int i=0;i<m;i++)
        matrix[i*width+0] = i * MISMATCH;   // matrix[i][0]
    for(int j=0;j<n;j++)
        matrix[0*width+j] = j * MISMATCH;   // matrix[0][j];

    for(int i=1;i<m;i++) {
        for(int j=1;j<n;j++) {
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
void cuda_backtrack(int m, int n, int seqIdx, int *matrix, int *spaceRow, int *spaceForOtherRow, int maxLength) {

    int width = maxLength + 1;

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
void cuda_msa(char *centerSeq, char *seqs, int *matrix, int *space, int *spaceForOther, size_t pitch, int maxLength) {
    int seqIdx = threadIdx.x + blockIdx.x * blockDim.x;

    int *matrixRow = (int*)((char*)matrix + seqIdx * pitch);
    char *seq = seqs + (maxLength+1) * seqIdx;

    int m = cuda_strlen(centerSeq);
    int n = cuda_strlen(seq);

    // 当前匹配的字符串所需要填的空格数组
    int *spaceRow = space + (seqIdx * (m+1));
    int *spaceForOtherRow = spaceForOther + (seqIdx * (maxLength+1));

    printf("centerSeq: %s, seq: %s\n", centerSeq, seq);
    printf("seqIdx: %d, m: %d, n: %d\n", seqIdx, m, n);

    cuda_nw(centerSeq, seq, matrixRow, maxLength);

    cuda_backtrack(m, n, seqIdx, matrixRow, spaceRow, spaceForOtherRow, maxLength);
    //printf("\n");
    //printMatrix(spaceRow, 1, m+1);
    //printMatrix(spaceForOtherRow, 1, n+1);
}

/**
 * 输出MSA
 * 设共有n条串，平均长度m
 * 构造中心串复杂度为:O(nm)
 * 构造其他条串复杂度为:O(nm)
 */
void output(int *space, int *spaceForOther) {
    vector<string> allAlignedStrs;
    int width = centerSeq.size() + 1;

    // 将所有串添加的空格汇总到一个数组中
    // 然后给中心串插入空格
    string alignedCenter(centerSeq);
    vector<int> spaceForCenter(centerSeq.size()+1, 0);
    for(int pos = centerSeq.size(); pos >= 0; pos--) {
        int count = 0;
        for(int idx = 0; idx < seqs.size(); idx++)
            count = (space[idx*width+pos] > count) ? space[idx*width+pos] : count;
        spaceForCenter[pos] = count;
        if(spaceForCenter[pos] > 0)
            //printf("pos:%d, space:%d\n", pos, spaceForCenter[pos]);
            alignedCenter.insert(pos, spaceForCenter[pos], '-');
    }

    printf("\n\n%s\n", alignedCenter.c_str());
    allAlignedStrs.push_back(alignedCenter);

    width = maxLength + 1;
    for(int idx = 0; idx < seqs.size(); idx++) {
        int shift = 0;
        string alignedStr(seqs[idx]);
        // 先插入自己比对时的空格
        for(int pos = seqs[idx].size(); pos >= 0; pos--) {
            if(spaceForOther[idx*width+pos] > 0)
                alignedStr.insert(pos, spaceForOther[idx*width+pos], '-');
        }
        // 再插入其他串比对时引入的空格
        for(int pos = 0; pos < spaceForCenter.size(); pos++) {
            int num = spaceForCenter[pos] - space[idx*width+pos];
            alignedStr.insert(pos+shift, num, '-');
            shift += spaceForCenter[pos];
        }
        printf("%s\n", alignedStr.c_str());
        allAlignedStrs.push_back(alignedStr);
    }

    // 将结果写入文件
    writeFastaFile("/home/wangchen/source/CUDA/CUDA-MSA/src/output2.fasta", allAlignedStrs);
}



void init() {
    // 读入所有字符串
    // centerSeq, 图中的纵向，决定了行数m
    // seqs[idx], 图中的横向，决定了列数n
    seqs = readFastaFile("/home/wangchen/source/CUDA/CUDA-MSA/test.fasta");
    centerSeq = seqs[0];
    seqs.erase(seqs.begin());

    maxLength = centerSeq.size();
    for(int i=0;i<seqs.size();i++)
        if( maxLength < seqs[i].size())
            maxLength = seqs[i].size();
    printf("max length: %d\n", maxLength);

    c_seqs = new char[(maxLength+1) * seqs.size()];
    for(int i=0;i<seqs.size();i++) {
        char *p = &(c_seqs[i * (maxLength + 1)]);
        strcpy(p, seqs[i].c_str());
    }
}


int main() {

    init();

    // 将二维数组一维化, 每行对应一条串
    int *d_space;
    int *d_spaceForOther;
    size_t pitch;
    int width = maxLength + 1;
    int height = seqs.size();
    cudaMalloc((void**)&d_space, height*(centerSeq.size()+1)*sizeof(int));
    cudaMalloc((void**)&d_spaceForOther, height*width*sizeof(int));
    cudaMemset(d_space, 0, height*(centerSeq.size()+1)*sizeof(int));
    cudaMemset(d_spaceForOther, 0, height*width*sizeof(int));

    // 每条串的DP矩阵是一行
    int *d_matrix;
    cudaMallocPitch((void**)&d_matrix, &pitch, width*(centerSeq.size()+1)*sizeof(int), height);

    // 给字符串分配空间
    char *d_centerSeq;
    cudaMalloc((void**)&d_centerSeq, (centerSeq.size()+1) * sizeof(char));
    cudaMemcpy(d_centerSeq, centerSeq.c_str(), (centerSeq.size()+1)*sizeof(char), cudaMemcpyHostToDevice);
    char *d_seqs;
    cudaMalloc((void**)&d_seqs, width*height*sizeof(char));
    cudaMemcpy(d_seqs, c_seqs, width*height*sizeof(char), cudaMemcpyHostToDevice);

    cuda_msa<<<1, seqs.size()>>>(d_centerSeq, d_seqs, d_matrix, d_space, d_spaceForOther, pitch, maxLength);


    int *space = new int[height*(centerSeq.size()+1)];
    int *spaceForOther = new int[height*width];
    cudaMemcpy(space, d_space, (centerSeq.size()+1)*height*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(spaceForOther, d_spaceForOther, width*height*sizeof(int), cudaMemcpyDeviceToHost);

    output(space, spaceForOther);

    cudaFree(d_space);
    cudaFree(d_spaceForOther);
    cudaFree(d_matrix);
    cudaFree(d_centerSeq);
    cudaFree(d_seqs);

    delete c_seqs;
    delete space;
    delete spaceForOther;

    return 0;
}
