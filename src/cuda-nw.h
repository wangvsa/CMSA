#ifndef _CUDA_NW_H_
#define _CUDA_NW_H_
#include <vector>
#include <string>

typedef struct DPCell_t {
    short score;
    short x_gap;
    short y_gap;
} DPCell;

void multi_gpu_msa(int workCount, std::string centerSeq, std::vector<std::string> seqs, int maxLength, short *space, short *spaceForOther);

//void cuda_msa(int workCount, std::string centerSeq, std::vector<std::string> seqs, int maxLength, short *space, short *spaceForOther);

#endif
