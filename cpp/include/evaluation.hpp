
#ifndef EVALUATION_HPP
#define EVALUATION_HPP

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

template <typename T> struct result_dict {
  typedef std::map<float, std::unordered_map<std::string, std::vector<T>>> type;
};
#endif // !EVALUATION_HPP
