#ifndef UTILS_HPP_
#define UTILS_HPP_

#include "common.hpp"

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems);
std::vector<std::string> split(const std::string &s, char delim);
std::vector<std::string> split(const std::string &s, char delim, double sample_prob);

#endif
