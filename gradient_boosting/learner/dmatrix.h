#ifndef LEARNER_DMATRIX_H_
#define LEARNER_DMATRIX_H_
#include <cstdlib> // NULL
#include "utils/stream.h" // FileStream
#include "utils/utils.h" // check
#include "data.h" // BoosterInfo

namespace gboost {
namespace learner {
struct MetaInfo {
  // Information needed by booster 
  // BoosterInfo does not implement save and load,
  // all serialization is done in MetaInfo
  BoosterInfo info;
  // Label of each instance
  std::vector<float> labels;
  // The index of begin and end of a group
  // needed when the learning task is ranking
  std::vector<bst_uint> group_ptr;
  // Weights of each instance, optional
  std::vector<float> weights;
  // Initialized margins,
  // if specified, gboost will start from this init margin
  // can be used to specify initial prediction to boost from
  std::vector<float> base_margin;
  // \brief version flag, used to check version of this info
  static const int kVersion = 0;

  inline void load_binary(utils::FileStream &fi) {
    int version;
    utils::check(fi.read(&version, sizeof(version)) != 0, "MetaInfo: invalid format");
    utils::check(fi.read(&info.num_row, sizeof(info.num_row)) != 0, "MetaInfo: invalid format");
    utils::check(fi.read(&info.num_col, sizeof(info.num_col)) != 0, "MetaInfo: invalid format");
    utils::check(fi.read(&labels), "MetaInfo: invalid format");
    utils::check(fi.read(&group_ptr), "MetaInfo: invalid format");
    utils::check(fi.read(&weights), "MetaInfo: invalid format");
    utils::check(fi.read(&info.root_index), "MetaInfo: invalid format");
    utils::check(fi.read(&base_margin), "MetaInfo: invalid format");
  }
  inline void save_binary(utils::FileStream &fo) const {
    int version = kVersion;
    fo.write(&version, sizeof(version));
    fo.write(&info.num_row, sizeof(info.num_row));
    fo.write(&info.num_col, sizeof(info.num_col));
    fo.write(labels);
    fo.write(group_ptr);
    fo.write(weights);
    fo.write(info.root_index);
    fo.write(base_margin);
  }
  // \brief clear all the information */
  inline void clear(void) {
    labels.clear();
    group_ptr.clear();
    weights.clear();
    info.root_index.clear();
    base_margin.clear();
    info.num_row = info.num_col = 0;
  }
};
struct DMatrix {
  const int magic;
  // Meta information about the dataset
  MetaInfo info;
  // Cache pointer to verify if the data structure is cached in some learner
  // used to verify if DMatrix is cached
  void *cache_learner_ptr_;
  // Default constructor
  explicit DMatrix(int magic): magic(magic), cache_learner_ptr_(NULL) {}
  virtual ~DMatrix() {}
};
}
}
#endif
