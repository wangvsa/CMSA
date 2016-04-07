#include <stdio.h>
#include "util.h"
#include "sp.h"
#include "center-star.h"
#include "cuda-nw.h"
#include "nw.h"
#include "omp.h"
#include "global.h"
using namespace std;


/**
 * 定义全局变量
 * centerSeq 存储中心串
 * seqs 存储所有其他串
 */
string centerSeq;
vector<string> seqs;
int maxLength;      // 最长的串的长度


/**
 * 从path读如fasta格式文件，
 * 完成初始化工作并输出相关信息
 */
void init(const char *path) {
    // 读入所有字符串
    // centerSeq, 图中的纵向，决定了行数m
    // seqs[idx], 图中的横向，决定了列数n
    seqs = readFastaFile(path);

    // 找出中心串
    int centerSeqIdx = findCenterSequence(seqs);

    centerSeq = seqs[0];
    seqs.erase(seqs.begin() + centerSeqIdx);

    unsigned long sumLength = 0;
    maxLength = centerSeq.size();
    int minLength = centerSeq.size();
    for(int i=0;i<seqs.size();i++) {
        sumLength += seqs[i].size();
        if( maxLength < seqs[i].size())
            maxLength = seqs[i].size();
        if( minLength > seqs[i].size())
            minLength = seqs[i].size();
    }
    int avgLength = sumLength / seqs.size();
    printf("sequences size: %d\n", seqs.size());
    printf("max length: %d, min length: %d, avg length: %d\n", maxLength, minLength, avgLength);
}

/**
  * 将MSA结果输出到path文件中
  * 共有n条串，平均长度m
  * 构造带空格的中心串复杂度为:O(nm)
  * 构造带空格的其他条串复杂度为:O(nm)
  */
void output(short *space, short *spaceForOther, const char* path) {
    vector<string> allAlignedStrs;

    int sWidth = centerSeq.size() + 1;      // space[] 的每条串宽度
    int soWidth = maxLength + 1;            // spaceForOther[] 的每条串宽度

    // 将所有串添加的空格汇总到一个数组中
    // 然后给中心串插入空格
    string alignedCenter(centerSeq);
    vector<int> spaceForCenter(centerSeq.size()+1, 0);
    for(int pos = centerSeq.size(); pos >= 0; pos--) {
        int count = 0;
        for(int idx = 0; idx < seqs.size(); idx++)
            count = (space[idx*sWidth+pos] > count) ? space[idx*sWidth+pos] : count;
        spaceForCenter[pos] = count;
        if(spaceForCenter[pos] > 0)
            //printf("pos:%d, space:%d\n", pos, spaceForCenter[pos]);
            alignedCenter.insert(pos, spaceForCenter[pos], '-');
    }

    //printf("\n\n%s\n", alignedCenter.c_str());
    allAlignedStrs.push_back(alignedCenter);

    for(int idx = 0; idx < seqs.size(); idx++) {
        int shift = 0;
        string alignedStr(seqs[idx]);
        // 先插入自己比对时的空格
        for(int pos = seqs[idx].size(); pos >= 0; pos--) {
            if(spaceForOther[idx*soWidth+pos] > 0)
                alignedStr.insert(pos, spaceForOther[idx*soWidth+pos], '-');
        }
        // 再插入其他串比对时引入的空格
        for(int pos = 0; pos < spaceForCenter.size(); pos++) {
            int num = spaceForCenter[pos] - space[idx*sWidth+pos];
            if(num > 0) {
                alignedStr.insert(pos+shift, num, '-');
            }
            shift += spaceForCenter[pos];
        }
        //printf("%s\n", alignedStr.c_str());
        allAlignedStrs.push_back(alignedStr);
    }

    // 将结果写入文件
    printf("write to the output file.\n");
    writeFastaFile(path, allAlignedStrs);
}


int main(int argc, char *argv[]) {

    int argvIdx = parseOptions(argc, argv);
    // 输入错误选项或选项少时不执行程序
    if(argvIdx < 0) return 0;

    const char *inputPath = argv[argvIdx];
    const char *outputPath = argv[argvIdx+1];

    // 读入所有串，找出中心串
    init( inputPath );

    // Host端的纪录空格的数组
    short *space = new short[seqs.size() * (centerSeq.size() + 1)];
    short *spaceForOther = new short[seqs.size() * (maxLength + 1)];


    // 根据用户需要运行的模式来分配工作量
    int height = seqs.size() / 2;
    if( MODE == GPU_ONLY )
        height = seqs.size();
    if( MODE == CPU_ONLY )
        height = 0;

    msa(BLOCKS, THREADS, maxLength, height, centerSeq, seqs, space, spaceForOther);
    cpu_msa(centerSeq, seqs, height, space, spaceForOther, maxLength);

    output(space, spaceForOther, outputPath);

    delete[] space;
    delete[] spaceForOther;
}


