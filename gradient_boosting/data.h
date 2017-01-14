#ifndef DATA_H_
#define DATA_H_
#include <vector>
namespace gboost {
// \brief unsigned interger type used in boost,
//        used for feature index and row index
typedef unsigned bst_uint;
// \brief float type, used for storing statistics
typedef float bst_float;

class IFMatrix {
};

struct SparseBatch {
  // \brief an entry of sparse vector
  struct Entry {
    // feature index
    bst_uint index;
    // feature value
    bst_float fvalue;
    // default constructor
    Entry(void) {}
    Entry(bst_uint index, bst_float fvalue) : index(index), fvalue(fvalue) {}
    // reversely compare feature values
    inline static bool CmpValue(const Entry &a, const Entry &b) {
      return a.fvalue < b.fvalue;
    }
  };
};

struct RowBatch: public SparseBatch {
};

// Extra information that might needed by gbm and tree module
// these information are not necessarily presented, and can be empty
struct BoosterInfo {
  // Number of rows in the data
  size_t num_row;
  // Number of columns in the data
  size_t num_col;
  // Specified root index of each instance,
  // can be used for multi task setting
  std::vector<unsigned> root_index;
  // Number of rows, number of columns
  BoosterInfo(void) : num_row(0), num_col(0) { }
};

// \brief read-only column batch, used to access columns,
//        the columns are not required to be continuous
struct ColBatch : public SparseBatch {
};

}
#endif
