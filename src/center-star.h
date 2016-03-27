#ifndef _CENTER_STAR_H_
#define _CENTER_STAR_H_

/**
 * 8个字符转换为1个整数
 * 8 char = 16 bit
 * 0 ~ 65535
 *
 * 'A' = 00
 * 'C' = 01
 * 'T' = 10
 * 'G' = 11
 */
int charsToIndex(const char *str);

void setOccVector(const char *str, int *vec);

int countSequences(const char *str, int *vec);
#endif
