#ifndef _UTIL_H_
#define _UTIL_H_
#include <string>
#include <vector>

std::vector<std::string> readFastaFile(const char *path);

void writeFastaFile(const char *path, std::vector<std::string> strs);

#endif
