#include <stdio.h>
#include "omp.h"
#include "nw.h"
#include "global.h"
using namespace std;


// 定义二维数组
typedef vector< vector<int> > Matrix;


void printMatrix(Matrix matrix) {
    int m = matrix.size();
    int n = matrix[0].size();
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

Matrix nw(string str1, string str2) {
    // m行, n列
    int m = str1.size() + 1;
    int n = str2.size() + 1;

    Matrix matrix(m, vector<int>(n));
    // 初始化矩阵
    for(int i=0;i<n;i++)
        matrix[0][i] = i * MISMATCH;
    for(int i=0;i<m;i++)
        matrix[i][0] = i * MISMATCH;

    for(int i=1;i<m;i++) {
        for(int j=1;j<n;j++) {
            int up = matrix[i-1][j] + GAP;
            int left = matrix[i][j-1] + GAP;
            int diag = matrix[i-1][j-1] + ((str1[i-1]==str2[j-1])?MATCH:MISMATCH);

            matrix[i][j] = cpu_max(up, left, diag);
        }
    }

    // printMatrix(matrix);
    return matrix;
}

void backtrack(Matrix matrix, string centerSeq, vector<string> seqs, int seqIdx, short *space, short *spaceForOther, int maxLength) {
    string str1 = centerSeq;
    string str2 = seqs[seqIdx];

    int m = str1.size();
    int n = str2.size();

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
}


void cpu_msa(string centerSeq, vector<string> seqs, int startIdx, short *space, short *spaceForOther, int maxLength) {

    if(startIdx >= seqs.size()) return;

    double start, end;

    // 计算DP矩阵, 执行backtrack
    Matrix matrix;
    #pragma omp parallel for private(matrix)
    for(int idx = startIdx; idx < seqs.size(); idx++) {
        matrix = nw(centerSeq, seqs[idx]);
        backtrack(matrix, centerSeq, seqs, idx, space, spaceForOther, maxLength);
        //printf("%d/%d, sequence length:%d\n", idx+1, seqs.size(), (int)seqs[idx].size());
    }

}
