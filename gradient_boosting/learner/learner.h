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
    }
    if (gbm_ == NULL || obj_ == NULL) {
      cfg_.push_back(std::make_pair(std::string(name), std::string(val)));
    }
  }
  inline void set_cache_data(const std::vector<DMatrix *> &mats) {}
  // \brief initialize the model
  inline void init_model(void) {
    // initialize model
    this->init_obj_gbm();
    // initialize GBM model
    gbm_->init_model();
  }
};

}
}
#endif
