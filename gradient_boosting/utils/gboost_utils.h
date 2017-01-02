#ifndef GBOOST_UTILS_H_
#define GBOOST_UTILS_H_
/**
 * \file gboost_utils.h
 * \brief simple utils to support the code
 */
#include <cstdio>
#include <cstdlib>

namespace gboost {
/** \brief namespace for helper utils of the project */
namespace utils {
inline void Error(const char *msg) {
  fprintf(stderr, "Error");
  exit(-1);
}

inline void Assert(bool exp) { if(!exp) Error( "AssertError" ); }

inline void Assert(bool exp, const char *msg ) { if(!exp) Error(msg); }

/** \brief replace fopen, report error the file open fails */
}
}

#endif
