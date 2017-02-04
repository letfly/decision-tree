#ifndef UTILS_CONFIG_H_
#define UTILS_CONFIG_H_
// \file gboost_config.h
// \brief helper class to load in configures from file
#define CRT_SECURE_NO_WARNINGS
#include <string>
#include "utils/utils.h" // assert, fopen_check

namespace gboost {
namespace utils {
class ConfigIterator{
 private:
  FILE *fi;
  char ch_buf;
  char s_name[256], s_buf[246], s_val[256];
  inline void skip_line() {
    do{
      ch_buf = fgetc(fi);
    } while (ch_buf!=EOF && ch_buf!='\n' && ch_buf!='\r');
  }
  inline void parse_str(char tok[]) {
    int i = 0;
    while ((ch_buf = fgetc(fi)) != EOF) {
      switch (ch_buf) {
        case '\\': tok[i++] = fgetc(fi); break;
        case '\"': tok[i++] = '\0'; return;
        case '\r':
        case '\n': error("unterminated string"); break;
        default: tok[i++] = ch_buf;
      }
    }
    error("unterminated string"); 
  }
  inline bool get_next_token(char tok[]){
    int i = 0;
    bool new_line = false;
    while (ch_buf != EOF) {
      switch (ch_buf) {
        case '#': skip_line(); new_line = true; break;
        case '\"':
          if (i == 0) {
            parse_str(tok);
            ch_buf = fgetc(fi);
            return new_line;
          } else error("token followed directly by string");
        case '=':
          if (i == 0) {
            ch_buf = fgetc(fi);
            tok[0] = '=';
            tok[1] = '\0';
          } else tok[i] = '\0';
          return new_line;
        case '\r':
        case '\n':
          if (i == 0) new_line = true;
        case '\t':
        case ' ' :
          ch_buf = fgetc(fi);
          if (i > 0) {
            tok[i] = '\0';
            return new_line;
          }
          break;
        default:
          tok[i++] = ch_buf;
          ch_buf = fgetc(fi);
          break;                    
      }
    }
    return true;
  }
 public:
  ConfigIterator(const char *fname) {
    fi = fopen_check(fname, "r");
    ch_buf = fgetc(fi);
  }
  // \brief destructor
  ~ConfigIterator() {}
  // \brief move iterator to next position
  // \return true if there is value in next position
  inline bool next(void) {
    while (!feof(fi)) {
      get_next_token(s_name);
      if (s_name[0] == '=') return false;
      if (get_next_token(s_buf) || s_buf[0]!='=') return false;
      if (get_next_token(s_val) || s_val[0]=='=') return false;
      return true;
    }
    return false;
  }
  // \brief get current name, called after next returns true
  // \return current parameter name
  inline const char *name(void)const { return s_name; }
  // \brief get current value, called after next returns true
  // \return current parameter value
  inline const char *val(void)const { return s_val; }
};
}
}
#endif
