#ifndef GBOOST_UTILS_H_
#define GBOOST_UTILS_H_
/**
 * \file gboost_utils.h
 * \brief simple utils to support the code
 */
#define CRT_SECURE_NO_WARNINGS
#ifndef MSC_VER
#define fopen64 fopen
#else

// use 64 bit offset, either to include this header in the beginning, or
#ifdef _FILE_OFFSET_BITS
#if _FILE_OFFSET_BITS == 32
#warning "FILE OFFSET BITS defined to be 32 bit"
#endif
#endif

#ifdef __APPLE__
#define off64_t off_t
#define fopen64 fopen
#endif

#define __FILE_OFFSET_BITS 64
extern "C" {
#include <sys/types.h>
}
#include <cstdio>
#endif

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

inline void Warning(const char *msg) { fprintf(stderr, "warning:%s\n", msg); }
/** \brief replace fopen, report error the file open fails */
inline FILE *FopenCheck(const char *fname, const char *flag) {
  FILE *fp = fopen64(fname, flag);
  if(fp == NULL) {
    fprintf(stderr, "can not open file \"%s\"\n", fname);
    exit(-1);
  }
  return fp;
}
}
}

#endif
