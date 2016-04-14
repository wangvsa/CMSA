#include <stdio.h>
#include "omp.h"
#include "nw.h"
#include "global.h"
using namespace std;


void printMatrix(short **matrix, int m, int n) {
    for(int i=0;i<m;i++) {
        for(int j=0;j<n;j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

int cpu_max(int v1, int v2) {
    return v1 > v2 ? v1 : v2;
}
int cpu_max(int v1, int v2, int v3) {
    return max(max(v1, v2), v3);
}

short** nw(string str1, string str2) {

    // m行, n列
    int m = str1.size();
    int n = str2.size();

    // 直接定义二维数组，比使用vector<vector>的形式节省内存
    // 缺点是需要自己管理内存释放
    short **matrix = new short*[m+1];
    for(int i = 0; i <= m; i++)
        matrix[i] = new short[n+1];

    // 初始化矩阵
    for(int j = 0; j <= n; j++)
        matrix[0][j] = j * MISMATCH;
    for(int i = 0; i <= m; i++)
        matrix[i][0] = i * MISMATCH;

    for(int i = 1; i <= m; i++) {
        for(int j = 1; j <= n; j++) {
            int up = matrix[i-1][j] + GAP;
            int left = matrix[i][j-1] + GAP;
            int diag = matrix[i-1][j-1] + ((str1[i-1]==str2[j-1])?MATCH:MISMATCH);

            matrix[i][j] = cpu_max(up, left, diag);
        }
    }

    // printMatrix(matrix, m, n);
    return matrix;
}

void backtrack(short **matrix, string centerSeq, string seq, int seqIdx, short *space, short *spaceForOther, int maxLength) {
    int m = centerSeq.size();
    int n = seq.size();

    int sWidth = m + 1;
    int soWidth = maxLength + 1;

    // 从(m, n) 遍历到 (0, 0)
    int i = m, j = n;           // DP矩阵的纬度是m+1, n+1
    while(i!=0 || j!=0) {
        int score = matrix[i][j];
        // printf("%d, %d\n", i, j);
        if(i > 0 && matrix[i-1][j] + GAP == score) {
            spaceForOther[seqIdx*soWidth+j]++;
            i--;
        } else if(j > 0 && matrix[i][j-1] + GAP == score) {
            space[seqIdx*sWidth+i]++;
            j--;
        } else {
            i--;
            j--;
        }
    }

    // 释放matrix[(m+1, n+1]内存
    for(int i = 0; i <= m; i++)
        delete[] matrix[i];
    delete[] matrix;
}


void cpu_msa(string centerSeq, vector<string> seqs, int startIdx, short *space, short *spaceForOther, int maxLength) {

    if(startIdx >= seqs.size()) return;

    double start, end;

    // 计算DP矩阵, 执行backtrack
    #pragma omp parallel for
    for(int idx = startIdx; idx < seqs.size(); idx++) {
        short **matrix = nw(centerSeq, seqs[idx]);
        backtrack(matrix, centerSeq, seqs[idx], idx, space, spaceForOther, maxLength);
        printf("%d/%lu, sequence length:%lu\n", idx+1, seqs.size(), seqs[idx].size());
    }

}
