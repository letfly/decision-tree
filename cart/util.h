#include <sstream> // stringstream

std::vector<int> inline range(int start, int stop=-1, int step=1) { // range() in main.cc and matrix.cc O(stop)
  std::vector<int> result;
  if (stop == -1) {
    stop = start;
    start = 0;
  }
  for (int i = start; i < stop; i += step) result.push_back(i);
  return result;
}

std::vector<std::string> inline split_string(const std::string &source, const char *delimiter = " ", bool keep_empty = false) { // split_string() in matrix.cc O(columns_size)
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

// 转换python的负切片索引至常规索引
template<typename T>
int inline negative_index_covert(std::vector<T> list, int index) {
  // 边界判断
  if (index >= 0) return index;
  else return index+list.size();
}

// 切片一个数组，不可变，类似python的[::]操作
template<typename T>
std::vector<T> inline slice(std::vector<T> list, int start=0, int end=-1, int step=1) {
  std::vector<T> result;
  start = negative_index_covert(list, start);
  end = negative_index_covert(list, end);
  for (int i = start; i < end; i +=step) {
    T element = list[i];
    result.push_back(element);
  }
  return result;
}
