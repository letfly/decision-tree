#ifndef UTILS_H_
#define UTILS_H_
#include <cstdlib> // exit

namespace gboost {
namespace utils {
inline void error(const char *msg) {
  fprintf(stderr, "%s\n", msg);
  exit(-1);
}
inline void assert(bool exp) { if (!exp) error("AssertError"); }
inline void assert(bool exp, const char *msg ) { if(!exp) error(msg); }

const int k_print_buffer = 1 << 22;
inline void check(bool exp, const char *fmt, ...) {
  if (!exp) {
    std::string msg(k_print_buffer, '\0');
    va_list args;
    va_start(args, fmt);
    vsnprintf(&msg[0], k_print_buffer, fmt, args);
    va_end(args);
    error(msg.c_str());
  }
}
inline FILE *fopen_check(const char *fname, const char *flag) {
  FILE *fp = fopen(fname, flag);
  check(fp != NULL, "can not open file \"%s\"\n", fname);
  return fp;
}

// \brief portable version of snprintf
inline int sprintf(char *buf, size_t size, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  int ret = vsnprintf(buf, size, fmt, args);
  va_end(args);
  return ret;
}
}

// easy utils that can be directly acessed in gboost
// \brief get the beginning address of a vector
template<typename T>
inline T *begin_ptr(std::vector<T> &vec) {
  if (vec.size() == 0) return NULL;
  else return &vec[0];
}
// \brief get the beginning address of a vector
template<typename T>
inline const T *begin_ptr(const std::vector<T> &vec) {
  if (vec.size() == 0) return NULL;
  else return &vec[0];
}

}

#endif
