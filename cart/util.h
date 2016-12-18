#include <sstream> // stringstream

std::vector<int> inline range(int start, int stop=-1, int step=1) { // range() in main.cc and matrix.cc
  std::vector<int> result;
  if (stop == -1) {
    stop = start;
    start = 0;
  }
  for (int i = start; i < stop; i += step) result.push_back(i);
  return result;
}

std::vector<std::string> inline split_string(const std::string &source, const char *delimiter = " ", bool keep_empty = false) { // split_string() in matrix.cc
  std::vector<std::string> results;

  size_t prev = 0;
  size_t next = 0;

  while ((next = source.find_first_of(delimiter, prev)) != std::string::npos) {
    if (keep_empty || (next-prev != 0)) results.push_back(source.substr(prev, next-prev));
    prev = next+1;
  }
  if (prev < source.size()) results.push_back(source.substr(prev));
  return results;
}

template<typename T>
std::string inline join(std::vector<T> &list, const char delimiter) { // join() in tree_node.cc
  std::stringstream ss;
  for (int i =0; i < list.size(); ++i) {
    ss << list[i];
    if (i < list.size()-1) ss << delimiter;
  }
  return ss.str();
}
