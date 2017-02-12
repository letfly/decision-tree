#ifndef LEARNER_LEARNER_H_
#define LEARNER_LEARNER_H_
#include <string>
#include "gbm/gbm.h" // IGradBooster, create_grad_booster
#include "learner/dmatrix.h" // DMatrix
#include "learner/objective.h" // IObjFunction, CreateObjFunction
#include "learner/evaluation.h" // EvalSet, size(

namespace gboost {
namespace learner {
class BoostLearner {
 private:
  // Gbm model that back everything
  gbm::IGradBooster *gbm_;
  // Name of gbm model used for training
  std::string name_gbm_;
  // Name of objective function
  std::string name_obj_;

  // objective fnction
  IObjFunction *obj_;
  // configurations
  std::vector< std::pair<std::string, std::string> > cfg_;
  // evaluation set
  EvalSet evaluator_;
  // maximum buffred row value
  float prob_buffer_row;

  // \brief initialize the objective function and GBM, 
  // if not yet done
  inline void init_obj_gbm(void) {
    if (obj_ != NULL) return;
    utils::assert(gbm_ == NULL, "GBM and obj should be NULL");
    obj_ = CreateObjFunction(name_obj_.c_str());
    gbm_ = gbm::create_grad_booster(name_gbm_.c_str());
    for (size_t i = 0; i < cfg_.size(); ++i) {
      obj_->set_param(cfg_[i].first.c_str(), cfg_[i].second.c_str());
      gbm_->set_param(cfg_[i].first.c_str(), cfg_[i].second.c_str());
    }
    if (evaluator_.size() == 0)
      evaluator_.add_eval(obj_->default_eval_metric());
  }

  // cache entry object that helps handle feature caching
  struct CacheEntry {
    const DMatrix *mat_;
    size_t buffer_offset_;
    size_t num_row_;
    CacheEntry(const DMatrix *mat, size_t buffer_offset, size_t num_row)
        :mat_(mat), buffer_offset_(buffer_offset), num_row_(num_row) {}
  };
  // data structure field
  // \brief the entries indicates that we have internal prediction cache
  std::vector<CacheEntry> cache_;
  // find internal bufer offset for certain matrix, if not exist, return -1
  inline int64_t find_buffer_offset(const DMatrix &mat) const {
    for (size_t i = 0; i < cache_.size(); ++i)
      if (cache_[i].mat_ == &mat && mat.cache_learner_ptr_ == this)
        if (cache_[i].num_row_ == mat.info.num_row())
          return static_cast<int64_t>(cache_[i].buffer_offset_);
    return -1;
  }
  // \brief training parameter for regression
  struct ModelParam{
    // \brief global bias
    float base_score;
    // \brief number of features
    unsigned num_feature;
    ModelParam(void) {
      base_score = 0.5f;
      num_feature = 0;
    }
    inline void set_param(const char *name, const char *val) {
      if (!strcmp("base_score", name)) base_score = static_cast<float>(atof(val));
      if (!strcmp("bst:num_feature", name)) num_feature = atoi(val);
    }
  };
  // model parameter
  ModelParam   mparam;
  // \brief get un-transformed prediction
  // \param data training data matrix
  // \param out_preds output vector that stores the prediction
  // \param ntree_limit limit number of trees used for boosted tree
  //   predictor, when it equals 0, this means we are using all the trees
  inline void predict_raw(const DMatrix &data,
                         std::vector<float> *out_preds,
                         unsigned ntree_limit = 0) const {
    gbm_->predict(data.fmat(), this->find_buffer_offset(data),
                  data.info.info, out_preds, ntree_limit);
    // add base margin
    std::vector<float> &preds = *out_preds;
    const bst_uint ndata = static_cast<bst_uint>(preds.size());
    if (data.info.base_margin.size() != 0) {
      utils::check(preds.size() == data.info.base_margin.size(),
                   "base_margin.size does not match with prediction size");
      for (bst_uint j = 0; j < ndata; ++j)
        preds[j] += data.info.base_margin[j];
    } else {
      for (bst_uint j = 0; j < ndata; ++j) {
        preds[j] += mparam.base_score;
      }
    }
  }
  // temporal storages for prediciton
  std::vector<float> preds_;
  // gradient pairs
  std::vector<bst_gpair> gpair_;
 public:
  BoostLearner() {
    obj_ = NULL;
    gbm_ = NULL;
  }
  ~BoostLearner() {
    if (obj_ != NULL) delete obj_;
    if (gbm_ != NULL) delete gbm_;
  }
  inline void set_param(const char *name, const char *val) {
    if (gbm_ == NULL) {
      if (!strcmp(name, "booster")) name_gbm_ = val;
      if (!strcmp(name, "objective")) name_obj_ = val;
      mparam.set_param(name, val);
    }
    if (gbm_ == NULL || obj_ == NULL) {
      cfg_.push_back(std::make_pair(std::string(name), std::string(val)));
    }
  }
  // \brief add internal cache space for mat, this can speedup prediction for matrix,
  //        please cache prediction for training and eval data
  //    warning: if the model is loaded from file from some previous training history
  //             set cache data must be called with exactly SAME 
  //             data matrices to continue training otherwise it will cause error
  // \param mats array of pointers to matrix whose prediction result need to be cached
  inline void set_cache_data(const std::vector<DMatrix *> &mats) {
    // estimate feature bound
    unsigned num_feature = 0;
    for (size_t i = 0; i < mats.size(); ++i) {
      bool dupilicate = false;
      for (size_t j = 0; j < i; ++j) {
        if (mats[i] == mats[j]) dupilicate = true;
      }
      if (dupilicate) continue;
      num_feature = std::max(num_feature, static_cast<unsigned>(mats[i]->info.num_col()));
    }
    char str_temp[25];
    if (num_feature > mparam.num_feature) {
      utils::sprintf(str_temp, sizeof(str_temp), "%u", num_feature);
      this->set_param("bst:num_feature", str_temp);
    }
  }
  // \brief initialize the model
  inline void init_model(void) {
    // initialize model
    this->init_obj_gbm();
    // initialize GBM model
    gbm_->init_model();
  }

  // Train()
  // \brief check if data matrix is ready to be used by training,
  //  if not intialize it
  // \param p_train pointer to the matrix used by training
  inline void check_init(DMatrix *p_train) {
    p_train->fmat()->init_col_access(prob_buffer_row);
  }
  // \brief update the model for one iteration
  // \param iter current iteration number
  // \param p_train pointer to the data matrix
  inline void update_one_iter(int iter, const DMatrix &train) {
    this->predict_raw(train, &preds_);
    obj_->get_gradient(preds_, train.info, iter, &gpair_);
    gbm_->do_boost(train.fmat(), train.info.info, &gpair_);
  }
  // \brief evaluate the model for specific iteration
  // \param iter iteration number
  // \param evals datas i want to evaluate
  // \param evname name of each dataset
  // \return a string corresponding to the evaluation result
  inline std::string eval_one_iter(int iter,
                                 const std::vector<const DMatrix*> &evals,
                                 const std::vector<std::string> &evname) {
    std::string res;
    char tmp[256];
    utils::sprintf(tmp, sizeof(tmp), "[%d]", iter);
    res = tmp;
    for (size_t i = 0; i < evals.size(); ++i) {
      this->predict_raw(*evals[i], &preds_);
      obj_->eval_transform(&preds_);
      res += evaluator_.eval(evname[i].c_str(), preds_, evals[i]->info);
    }
    return res;
  }
};

}
}
#endif
