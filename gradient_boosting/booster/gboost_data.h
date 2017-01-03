namespace gboost {
namespace booster {
typedef int bst_int;
typedef unsigned bst_uint;
typedef float bst_float;

class FMatrixS {
 private:
  std::vector<size_t> row_ptr;
 public:
  struct Line {
    bst_int len;
    const bst_uint *findex;
  };
  inline Line operator[] (size_t sidx) const {}
  inline void Clear(void) {}
  inline size_t AddRow(const std::vector<bst_uint> &findex,
                       const std::vector<bst_float> &fvalue) {}
  inline size_t NumRow(void) const { return row_ptr.size()-1; }
  inline size_t NumEntry(void) const{}
};
}
}
