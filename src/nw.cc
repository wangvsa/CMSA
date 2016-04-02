#include <stdio.h>
#include <string>
#include <vector>
#include "util.h"
#include "omp.h"
using namespace std;

#define MISMATCH -1
#define MATCH 0
#define GAP -1


// 定义二维数组
typedef vector< vector<int> > Matrix;

void backtrack(Matrix matrix, int seqIdx);

// space[i][j] 第i条串在中心串的j位置处插入的空格数
vector< vector<int> > space;
// 比对时自己插入的空格位置与个数
vector< vector<int> > spaceForOther;
int start[2];

string centerSeq;
vector<string> seqs;


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

int max(int v1, int v2) {
    return v1 > v2 ? v1 : v2;
}
int max(int v1, int v2, int v3) {
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

            matrix[i][j] = max(up, left, diag);
        }
    }

    // printMatrix(matrix);
    return matrix;
}

void backtrack(Matrix matrix, int seqIdx) {
    string str1 = centerSeq;
    string str2 = seqs[seqIdx];

    int m = str1.size();
    int n = str2.size();
    int len = max(m, n);

    //string alignedStr1, alignedStr2;

    // 从(m, n) 遍历到 (0, 0)
    int i = m, j = n;           // DP矩阵的纬度是m+1, n+1
    while(i!=0 || j!=0) {
        int score = matrix[i][j];
        // printf("%d, %d\n", i, j);
        if(i > 0 && matrix[i-1][j] + GAP == score) {
            spaceForOther[seqIdx][j]++;
            //alignedStr1 += str1[i-1];
            //alignedStr2 += "-";
            i--;
        } else if(j > 0 && matrix[i][j-1] + GAP == score) {
            space[seqIdx][i]++;
            //alignedStr1 += "-";
            //alignedStr2 += str2[j-1];
            j--;
        } else {
            //alignedStr1 += str1[i-1];
            //alignedStr2 += str2[j-1];
            i--;
            j--;
        }
    }

    //reverse(alignedStr1.begin(), alignedStr1.end());
    //reverse(alignedStr2.begin(), alignedStr2.end());
    //printf("%s\n%s\n", alignedStr1.c_str(), alignedStr2.c_str());
}

/**
 * 输出MSA
 * 设共有n条串，平均长度m
 * 构造中心串复杂度为:O(nm)
 * 构造其他条串复杂度为:O(nm)
 */
void output() {
    vector<string> allAlignedStrs;

    // 将所有串添加的空格汇总到一个数组中
    // 然后给中心串插入空格
    string alignedCenter(centerSeq);
    vector<int> spaceForCenter(centerSeq.size()+1, 0);
    for(int pos = centerSeq.size(); pos >= 0; pos--) {
        int count = 0;
        for(int idx = 0; idx < seqs.size(); idx++)
            count = (space[idx][pos] > count) ? space[idx][pos] : count;
        spaceForCenter[pos] = count;
        if(spaceForCenter[pos] > 0)
            alignedCenter.insert(pos, spaceForCenter[pos], '-');
    }

    //printf("\n\n%s\n", alignedCenter.c_str());
    allAlignedStrs.push_back(alignedCenter);

    for(int idx = 0; idx < seqs.size(); idx++) {
        int shift = 0;
        string alignedStr(seqs[idx]);
        // 先插入自己比对时的空格
        for(int pos = seqs[idx].size(); pos >= 0; pos--) {
            if(spaceForOther[idx][pos] > 0)
                alignedStr.insert(pos, spaceForOther[idx][pos], '-');
        }
        // 再插入其他串比对时引入的空格
        for(int pos = 0; pos < spaceForCenter.size(); pos++) {
            int num = spaceForCenter[pos] - space[idx][pos];
            alignedStr.insert(pos+shift, num, '-');
            shift += spaceForCenter[pos];
        }
        //printf("%s\n", alignedStr.c_str());
        allAlignedStrs.push_back(alignedStr);
    }

    // 将结果写入文件
    writeFastaFile("/home/hadoop/source/CUDA-MSA/src/output.fasta", allAlignedStrs);
}

void init() {
    // 读入所有字符串
    // centerSeq, 图中的纵向，决定了行数m
    // seqs[idx], 图中的横向，决定了列数n
    seqs = readFastaFile("/home/hadoop/source/CUDA-MSA/test.fasta");
    centerSeq = seqs[0];
    seqs.erase(seqs.begin());

    // 设置空格数组
    for(int idx = 0; idx < seqs.size(); idx++) {
        vector<int> tmp1(centerSeq.size()+1, 0);
        space.push_back(tmp1);
        vector<int> tmp2(seqs[idx].size()+1, 0);
        spaceForOther.push_back(tmp2);
    }
}


int main() {

    double start, end;

    // 1. 读入所有串, 初始化相关矩阵
    start = omp_get_wtime();
    init();
    end = omp_get_wtime();
    printf("Init, use time:%f\n",end-start);

    // 2. 计算DP矩阵, 执行backtrack
    Matrix matrix;
    start = omp_get_wtime();
    #pragma omp parallel for private(matrix)
    for(int idx = 0; idx < seqs.size(); idx++) {
        matrix = nw(centerSeq, seqs[idx]);
        backtrack(matrix, idx);
        printf("%d/%d, sequence length:%d\n", idx+1, seqs.size(), (int)seqs[idx].size());
    }
    end = omp_get_wtime();
    printf("DP calculation, use time:%f\n", end-start);

    // 3. 将结果写入到文件
    start = omp_get_wtime();
    output();
    end = omp_get_wtime();
    printf("Output, use time:%f\n", end-start);

    return 0;
}
