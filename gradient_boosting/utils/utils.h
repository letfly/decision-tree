#ifndef UTILS_H_
#define UTILS_H_
#include <cstdlib> // exit

namespace gboost {
namespace utils {
inline void error(const char *msg) {
  fprintf(stderr, "Error");
  exit(-1);
}
inline void assert(bool exp) { if (!exp) error("AssertError"); }
inline void assert(bool exp, const char *msg ) { if(!exp) error(msg); }

inline FILE *fopen_check(const char *fname, const char *flag) {
  FILE *fp = fopen(fname, flag);
  if (fp == NULL) {
    fprintf(stderr, "can not open file \"%s\"\n", fname);
    exit(-1);
  }
  return fp;
}
}
}

#endif
