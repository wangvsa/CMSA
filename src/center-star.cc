#include <stdio.h>
#include <bitset>
#include <string.h>
#include "center-star.h"
using namespace std;


int charsToIndex(const char *str) {
    bitset<16> bits(0x0000);
    for(int i=0;i<8;i++) {
        switch(str[i]) {
            case 'A':       // 00
                break;
            case 'C':       // 01
                bits[i*2+1] = 1;
                break;
            case 'T':       // 10
                bits[i*2] = 1;
                break;
            case 'G':       // 11
                bits[i*2] = 1;
                bits[i*2+1] = 1;
                break;
        }
    }
    return (int) (bits.to_ulong());
}

/**
 * 一条串中的每个索引最多只能增加一次
 * 使用一个额外的bool[65536]来纪录是否已经加过一次
 */
void setOccVector(const char *str, int *vec) {
    bool flag[65536] = {false};

    int len = strlen(str);
    int n = len / 8;
    for(int i=0;i<n;i++) {
        int index = charsToIndex(str+i*8);
        if(!flag[index]) {
            vec[index]++;
            flag[index] = true;
        }
    }
}


/**
 * 查询一条串中的每一段在其他串中出现的次数
 * 出现最多的一条串作为中心串
 */
int countSequences(const char *str, int *vec) {
    int len = strlen(str);
    int n = len / 8;
    int count = 0;
    for(int i=0;i<n;i++) {
        int index = charsToIndex(str+i*8);
        count += vec[index];
    }

    return count;
}

