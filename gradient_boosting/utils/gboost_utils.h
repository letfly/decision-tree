#ifndef GBOOST_UTILS_H_
#define GBOOST_UTILS_H_
/**
 * \file gboost_utils.h
 * \brief simple utils to support the code
 */
namespace gboost {
/** \brief namespace for helper utils of the project */
namespace utils {
inline FILE *fopen_check(const char *fname, const char *flag) {
  FILE *fp = fopen(fname, flag);
  if (fp == NULL) {
    fprintf(stderr, "can not open file \"%s\"\n", fname);
    exit(-1);
  }
  return fp;
}
inline void Error(const char *msg) {
  fprintf(stderr, "Error");
  exit(-1);
}

inline void assert(bool exp) { if (!exp) Error("AssertError"); }

inline void assert(bool exp, const char *msg ) { if(!exp) Error(msg); }

/** \brief replace fopen, report error the file open fails */
}
}

#endif
