#ifndef IO_SIMPLE_FMATRIX_H_
#define IO_SIMPLE_FMATRIX_H_
#include "data.h" // IFMatrix, RowBatch
#include "utils/random.h" // sample_binary
#include "utils/iterator.h" // IIterator
#include "utils/stream.h" // FileStream
#include "utils/utils.h" // check, begin_ptr

namespace gboost {
namespace io {
template<typename IndexType>
class SparseCSRMBuilder {
 private:
  // \brief pointer to each of the row
  std::vector<size_t> &rptr;
  // \brief index of nonzero entries in each row
  std::vector<IndexType> &findex;
 public:
  SparseCSRMBuilder(std::vector<size_t> &p_rptr,
                    std::vector<IndexType> &p_findex)
      :rptr(p_rptr), findex(p_findex) {
    utils::assert(!false, "enabling bug");
  }
  // \brief step 1: initialize the number of rows in the data, not necessary exact
  // \nrows number of rows in the matrix, can be smaller than expected
  inline void init_budget(size_t nrows = 0) {
    rptr.clear();
    rptr.resize(nrows + 1, 0);
  }
  // \brief step 2: add budget to each rows, this function is called when aclist is used
  // \param row_id the id of the row
  // \param nelem  number of element budget add to this row
  inline void add_budget(size_t row_id, size_t nelem = 1) {
    if (rptr.size() < row_id + 2) rptr.resize(row_id + 2, 0);
    rptr[row_id + 1] += nelem;
  }
  // \brief step 3: initialize the necessary storage
  inline void init_storage(void) {
    // initialize rptr to be beginning of each segment
    size_t start = 0;
    for (size_t i = 1; i < rptr.size(); i++) {
      size_t rlen = rptr[i];
      rptr[i] = start;
      start += rlen;
    }
    findex.resize(start);
  }
  // \brief step 4:
  // used in indicator matrix construction, add new
  // element to each row, the number of calls shall be exactly same as add_budget
  inline void push_elem(size_t row_id, IndexType col_id) {
    size_t &rp = rptr[row_id + 1];
    findex[rp++] = col_id;
  }
};

class FMatrixS: public IFMatrix {
 private:
  // \brief list of row index that are buffered
  std::vector<bst_uint> buffered_rowset_;
  // \brief column pointer of CSC format
  std::vector<size_t> col_ptr_;
  // \brief column datas in CSC format
  std::vector<ColBatch::Entry> col_data_;

  // row iterator
  utils::IIterator<RowBatch> *iter_;
  // \return whether column access is enabled
  virtual bool have_col_access(void) const {
    return col_ptr_.size() != 0;
  }
  // \brief get number of colmuns
  virtual size_t num_col(void) const {
    utils::check(this->have_col_access(), "num_col:need column access");
    return col_ptr_.size() - 1;
  }
  // \brief get the row iterator associated with FMatrix
  virtual utils::IIterator<RowBatch> *row_iterator(void) {
    iter_->before_first();
    return iter_;
  }
  // one batch iterator that return content in the matrix
  struct OneBatchIter: utils::IIterator<ColBatch> {
    // whether is at first
    bool at_first_;
    OneBatchIter(void): at_first_(true) {}
    virtual bool next(void) {
      if (!at_first_) return false;
      at_first_ = false;
      return true;
    }
    // temporal space for batch
    ColBatch batch_;
    virtual const ColBatch &value(void) const { return batch_; }
    virtual void before_first(void) { at_first_ = true; }
    // data content
    std::vector<bst_uint> col_index_;
    std::vector<ColBatch::Inst> col_data_;
    inline void set_batch(const std::vector<size_t> &ptr,
                         const std::vector<ColBatch::Entry> &data) {
      batch_.size = col_index_.size();
      col_data_.resize(col_index_.size(), SparseBatch::Inst(NULL,0));
      for (size_t i = 0; i < col_data_.size(); ++i) {
        const bst_uint ridx = col_index_[i];
        col_data_[i] = SparseBatch::Inst(&data[0] + ptr[ridx],
                                         static_cast<bst_uint>(ptr[ridx+1] - ptr[ridx]));
      }
      batch_.col_index = utils::begin_ptr(col_index_);
      batch_.col_data = utils::begin_ptr(col_data_);
      this->before_first();
    }
  };
  // column iterator
  OneBatchIter col_iter_;
  // \brief get the column based  iterator
  virtual utils::IIterator<ColBatch> *col_iterator(void) {
    size_t ncol = this->num_col();
    col_iter_.col_index_.resize(ncol);
    for (size_t i = 0; i < ncol; ++i)
      col_iter_.col_index_[i] = static_cast<bst_uint>(i);
    col_iter_.set_batch(col_ptr_, col_data_);
    return &col_iter_;
  }
  // \brief get number of buffered rows
  virtual const std::vector<bst_uint> &buffered_rowset(void) const {
    return buffered_rowset_;
  }
  // \brief get column size
  virtual size_t get_col_size(size_t cidx) const {
    return col_ptr_[cidx+1] - col_ptr_[cidx];
  }
  // \brief colmun based iterator
  virtual utils::IIterator<ColBatch> *col_iterator(const std::vector<bst_uint> &fset) {
    col_iter_.col_index_ = fset;
    col_iter_.set_batch(col_ptr_, col_data_);
    return &col_iter_;
  }
 public:
  FMatrixS(utils::IIterator<RowBatch> *iter) { this->iter_ = iter; }
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
    utils::check(fi.read(utils::begin_ptr(*out_ptr), out_ptr->size() * sizeof(size_t)) != 0,
                  "invalid input file format");
    out_data->resize(out_ptr->back());
    if (out_data->size() != 0) {
      utils::assert(fi.read(utils::begin_ptr(*out_data), out_data->size() * sizeof(RowBatch::Entry)) != 0,
                    "invalid input file format");
    }
  }
  // \brief load column access data from stream
  // \param fo output stream to load from
  inline void load_col_access(utils::FileStream &fi) {
    utils::check(fi.read(&buffered_rowset_), "invalid input file format");
    if (buffered_rowset_.size() != 0) {
      this->load_binary(fi, &col_ptr_, &col_data_);
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
    fo.write(utils::begin_ptr(ptr), ptr.size() * sizeof(size_t));
    if (data.size() != 0)
        fo.write(utils::begin_ptr(data), data.size() * sizeof(RowBatch::Entry));
  }
  // \brief save column access data into stream
  // \param fo output stream to save to
  inline void save_col_access(utils::FileStream &fo) const {
    fo.write(buffered_rowset_);
    if (buffered_rowset_.size() != 0) {
      this->save_binary(fo, col_ptr_, col_data_);
    }
  }

  // \brief intialize column data
  // \param pkeep probability to keep a row
  inline void init_col_data(float pkeep) {
    buffered_rowset_.clear();
    // note: this part of code is serial, todo, parallelize this transformer
    SparseCSRMBuilder<RowBatch::Entry> builder(col_ptr_, col_data_);
    builder.init_budget(0);
    // start working
    iter_->before_first();
    while (iter_->next()) {
      const RowBatch &batch = iter_->value();
      for (size_t i = 0; i < batch.size; ++i) {
        if (pkeep == 1.0f || random::sample_binary(pkeep)) {
          buffered_rowset_.push_back(static_cast<bst_uint>(batch.base_rowid+i));
          RowBatch::Inst inst = batch[i];
          for (bst_uint j = 0; j < inst.length; ++j)
            builder.add_budget(inst[j].index);
        }
      }
    }
    builder.init_storage();

    iter_->before_first();
    size_t ktop = 0;
    while (iter_->next()) {
      const RowBatch &batch = iter_->value();
      for (size_t i = 0; i < batch.size; ++i) {
        if (ktop < buffered_rowset_.size() &&
            buffered_rowset_[ktop] == batch.base_rowid+i) {
          ++ktop;
          RowBatch::Inst inst = batch[i];
          for (bst_uint j = 0; j < inst.length; ++j)
            builder.push_elem(inst[j].index,
                              SparseBatch::Entry((bst_uint)(batch.base_rowid+i),
                                   inst[j].fvalue));
        }
      }
    }
    // sort columns
    bst_uint ncol = static_cast<bst_uint>(this->num_col());
    for (bst_uint i = 0; i < ncol; ++i)
      std::sort(&col_data_[0] + col_ptr_[i],
                &col_data_[0] + col_ptr_[i + 1], SparseBatch::Entry::cmp_value);
  }
  virtual void init_col_access(float pkeep = 1.0f) {
    if (this->have_col_access()) return;
    this->init_col_data(pkeep);
  }
  // \brief get column density
  virtual float get_col_density(size_t cidx) const {
    size_t nmiss = buffered_rowset_.size() - (col_ptr_[cidx+1] - col_ptr_[cidx]);
    return 1.0f - (static_cast<float>(nmiss)) / buffered_rowset_.size();
  }
};
}
}
#endif
