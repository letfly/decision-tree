#ifndef UTILS_RANDOM_H_
#define UTILS_RANDOM_H_
#include <cmath>
#include <cstdlib>

namespace gboost {
namespace random {
// \brief  return 1 with probability p, coin flip
inline double uniform() {
  srand(0);
  return static_cast<double>(rand())/(static_cast<double>(RAND_MAX)+1.0);
}
// \brief  return 1 with probability p, coin flip
inline int sample_binary(double p) {
  return uniform() < p;
}
// \brief return a random number in n
inline uint32_t next_uint32(uint32_t n) {
  return (uint32_t)std::floor(uniform() * n);
}
template<typename T>
inline void shuffle(T *data, size_t sz) {
  if (sz == 0) return;
  for (uint32_t i = (uint32_t)sz - 1; i > 0; i--) {
    std::swap(data[i], data[next_uint32(i + 1)]);
  }
}
// random shuffle the data inside, require PRNG
template<typename T>
inline void shuffle(std::vector<T> &data) {
  shuffle(&data[0], data.size());
}

}
}
#endif
