#ifndef IO_SIMPLE_DMATRIX_H_
#define IO_SIMPLE_DMATRIX_H_
#include <cstdio> // printf
#include <cstring> // strlen, strcmp
#include "data.h"
#include "io/simple_fmatrix.h" // FMatrixS
#include "learner/dmatrix.h" // info
#include "utils/iterator.h" // IIterator
#include "utils/stream.h" // FileStream
#include "utils/utils.h" // check, sprintf

namespace gboost {
namespace io {
class DMatrixSimple: public learner::DMatrix {
 private:
  FMatrixS *fmat_;

  struct OneBatchIter: utils::IIterator<RowBatch> {
    bool at_first_;
    // Pointer to parient
    DMatrixSimple *parent_;
    explicit OneBatchIter(DMatrixSimple *parent)
      : at_first_(true), parent_(parent) {}
    virtual ~OneBatchIter() {}
    virtual void before_first(void) { at_first_ = true; }

    // temporal space for batch
    RowBatch batch_;
    virtual bool next(void) {
      if (!at_first_) return false;
      at_first_ = false;
      batch_.size = parent_->row_ptr_.size() - 1;
      batch_.base_rowid = 0;
      batch_.ind_ptr = utils::begin_ptr(parent_->row_ptr_);
      batch_.data_ptr = utils::begin_ptr(parent_->row_data_);
      return true;
    }
    virtual const RowBatch &value(void) const { return batch_; }
  };

  // data fields
  // \brief row pointer of CSR sparse storage
  std::vector<size_t> row_ptr_;
  // \brief data in the row
  std::vector<RowBatch::Entry> row_data_;

  virtual IFMatrix *fmat(void) const { return fmat_; }
 public:
  static const int kMagic = 0xffffab01;
  // Constructor
  DMatrixSimple(): learner::DMatrix(kMagic) {
    fmat_ = new FMatrixS(new OneBatchIter(this));
    this->clear();
  }
  // \brief clear the storage */
  inline void clear(void) {
    row_ptr_.clear();
    row_ptr_.push_back(0);
    row_data_.clear();
    info.clear();
  }

  //
  inline bool load_binary(const char* fname, bool silent = false) {
    std::FILE *fp = fopen(fname, "rb");
    if (fp == NULL) return false;
    utils::FileStream fs(fp);
    this->load_binary(fs, silent, fname);
    fs.close();
    return true;
  }
  //
  inline void load_binary(utils::FileStream &fs, bool silent=false, const char *fname=NULL) {
    int tmagic;
    utils::check(fs.read(&tmagic, sizeof(tmagic))!=0, "invalid input file format");

    info.load_binary(fs);
    FMatrixS::load_binary(fs, &row_ptr_, &row_data_);
    fmat_->load_col_access(fs);
  }
  //
  inline void save_binary(const char* fname, bool silent = false) const {
    utils::FileStream fs(utils::fopen_check(fname, "wb"));
    int tmagic = kMagic;
    fs.write(&tmagic, sizeof(tmagic));

    info.save_binary(fs);
    FMatrixS::save_binary(fs, row_ptr_, row_data_);
    fmat_->save_col_access(fs);
    fs.close();
  }
  // \brief add a row to the matrix
  // \param feats features
  // \return the index of added row
  inline size_t add_row(const std::vector<RowBatch::Entry> &feats) {
    for (size_t i = 0; i < feats.size(); ++i) {
      row_data_.push_back(feats[i]);
      info.info.num_col = std::max(info.info.num_col, static_cast<size_t>(feats[i].index+1));
    }
    row_ptr_.push_back(row_ptr_.back() + feats.size());
    info.info.num_row += 1;
    return row_ptr_.size() - 2;
  }
  // \brief load from text file
  // \param fname name of text data
  // \param silent whether print information or not
  inline void load_text(const char* fname, bool silent = false) {
    this->clear();
    FILE* file = utils::fopen_check(fname, "r");
    float label; bool init = true;
    char tmp[1024];
    std::vector<RowBatch::Entry> feats;
    while (fscanf(file, "%s", tmp) == 1) {
      RowBatch::Entry e;
      if (sscanf(tmp, "%u:%f", &e.index, &e.fvalue) == 2) {
        feats.push_back(e);
      } else {
        if (!init) {
          info.labels.push_back(label);
          this->add_row(feats);
        }
        feats.clear();
        utils::check(sscanf(tmp, "%f", &label) == 1, "invalid LibSVM format");
        init = false;
      }
    }

    info.labels.push_back(label);
    this->add_row(feats);
  }

  // If binary buffer exists, it will reads from binary buffer, otherwise,
  // it will load from text file, and try to create a buffer file
  void cache_load(const char *fname, bool silent=false, bool savebuffer=true) {
    char bname[1024];
    // Create buffer
    utils::sprintf(bname, sizeof(bname), "%s.buffer", fname);
    if (!this->load_binary(bname, silent)) {
      this->load_text(fname, silent);
      if (savebuffer) this->save_binary(bname, silent);
    }
  }
};
}
}
#endif
