#ifndef _CUDA_NW_H_
#define _CUDA_NW_H_
#include <vector>
#include <string>

typedef struct DPCell_t {
    short score;
    short x_gap;
    short y_gap;
} DPCell;


/**
 * workCount    负责计算的序列数目
 * centerSeq    中心串
 * seqs         除中心串外的所有串
 * maxLength    最长串的长度
 */
void cuda_msa(int workCount, std::string centerSeq, std::vector<std::string> seqs, int maxLength, short *space, short *spaceForOther);

#endif
