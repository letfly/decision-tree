#ifndef IO_SIMPLE_FMATRIX_H_
#define IO_SIMPLE_FMATRIX_H_
#include "data.h" // IFMatrix
#include "utils/iterator.h" // IIterator
#include "utils/stream.h" // FileStream
#include "utils/utils.h" // check, begin_ptr

namespace gboost {
namespace io {
class FMatrixS: public IFMatrix {
 private:
  // \brief list of row index that are buffered
  std::vector<bst_uint> buffered_rowset_;
  // \brief column pointer of CSC format
  std::vector<size_t> col_ptr_;
  // \brief column datas in CSC format
  std::vector<ColBatch::Entry> col_data_;

 public:
  FMatrixS(utils::IIterator<RowBatch> *iter) {}
  // \brief load data from binary stream
  // \param fi input stream
  // \param out_ptr pointer data
  // \param out_data data content
  inline static void load_binary(utils::FileStream &fi,
                                std::vector<size_t> *out_ptr,
                                std::vector<RowBatch::Entry> *out_data) {
    size_t nrow;
    utils::check(fi.read(&nrow, sizeof(size_t)) != 0, "invalid input file format");
    out_ptr->resize(nrow + 1);
    utils::check(fi.read(begin_ptr(*out_ptr), out_ptr->size() * sizeof(size_t)) != 0,
                  "invalid input file format");
    out_data->resize(out_ptr->back());
    if (out_data->size() != 0) {
      utils::assert(fi.read(begin_ptr(*out_data), out_data->size() * sizeof(RowBatch::Entry)) != 0,
                    "invalid input file format");
    }
  }
  // \brief load column access data from stream
  // \param fo output stream to load from
  inline void load_col_access(utils::FileStream &fi) {
    utils::check(fi.read(&buffered_rowset_), "invalid input file format");
    if (buffered_rowset_.size() != 0) {
      load_binary(fi, &col_ptr_, &col_data_);
    }
  }

  // \brief save data to binary stream
  // \param fo output stream
  // \param ptr pointer data
  // \param data data content
  inline static void save_binary(utils::FileStream &fo,
                                const std::vector<size_t> &ptr,
                                const std::vector<RowBatch::Entry> &data) {
    size_t nrow = ptr.size() - 1;
    fo.write(&nrow, sizeof(size_t));
    fo.write(begin_ptr(ptr), ptr.size() * sizeof(size_t));
    if (data.size() != 0) {
      fo.write(begin_ptr(data), data.size() * sizeof(RowBatch::Entry));
    }
  }
  // \brief save column access data into stream
  // \param fo output stream to save to
  inline void save_col_access(utils::FileStream &fo) const {
    fo.write(buffered_rowset_);
    if (buffered_rowset_.size() != 0) {
      save_binary(fo, col_ptr_, col_data_);
    }
  }
};
}
}
#endif
