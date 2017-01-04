#ifndef GBOOST_DATA_H_
#define GBOOST_DATA_H_
#include "utils/gboost_utils.h" // assert
namespace gboost {
namespace booster {
typedef int bst_int;
typedef unsigned bst_uint;
typedef float bst_float;
const bool bst_debug = false;

class FMatrixS {
 private:
  std::vector<size_t> row_ptr;
  std::vector<bst_uint> findex;
  std::vector<bst_float> fvalue;
 public:
  struct Line {
    bst_int len;
    const bst_uint *findex;
    const bst_float *fvalue;
  };
  inline size_t num_row(void) const { return row_ptr.size()-1; }
  inline Line operator[] (size_t sidx) const {
    Line sp;
    utils::assert(!bst_debug || sidx<this->num_row(), "row id exceed bound");
    sp.len = static_cast<bst_uint>(row_ptr[sidx+1] - row_ptr[sidx]);
    sp.findex = &findex[row_ptr[sidx]];
    sp.fvalue = &fvalue[row_ptr[sidx]];
    return sp;
  }
  inline void clear(void) {
    row_ptr.resize(0);
    findex.resize(0);
    fvalue.resize(0);
    row_ptr.push_back(0);
  }
  inline size_t add_row(const Line &feat, unsigned fstart = 0, unsigned fend = UINT_MAX) {
    utils::assert(feat.len >= 0, "sparse feature length can not be negative");
    unsigned cnt = 0;
    for (unsigned i = 0; i < feat.len; ++i) {
      if (feat.findex[i]<fstart || feat.findex[i]>=fend) continue;
      findex.push_back(feat.findex[i]);
      fvalue.push_back(feat.fvalue[i]);
      ++cnt;
    }
    row_ptr.push_back(row_ptr.back()+cnt);
    return row_ptr.size()-2;
  }
  inline size_t add_row(const std::vector<bst_uint> &findex,
                        const std::vector<bst_float> &fvalue) {
    FMatrixS::Line l;
    utils::assert(findex.size() == fvalue.size());
    l.findex = &findex[0];
    l.fvalue = &fvalue[0];
    l.len = static_cast<bst_uint>(findex.size());
    return this->add_row(l);
  }
  inline size_t num_entry(void) const { return findex.size(); }
};
}
}
#endif
