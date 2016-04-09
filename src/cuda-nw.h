#ifndef _CUDA_NW_H_
#define _CUDA_NW_H_
#include <vector>
#include <string>

/**
 * height 在GPU上计算的串数
 */
void cuda_msa(int BLOCKS, int THREADS, int maxLength, int height, std::string centerSeq, std::vector<std::string> seqs, short *space, short *spaceForOther);

#endif
