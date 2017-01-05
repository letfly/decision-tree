#ifndef LEARNER_LEARNER_H_
#define LEARNER_LEARNER_H_
#include <string>
#include "gbm/gbm.h"

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
 public:
  inline void set_param(const char *name, const char *val) {
    if (gbm_ == NULL) {
      if (!strcmp(name, "booster")) name_gbm_ = val;
      if (!strcmp(name, "objective")) name_obj_ = val;
    }
  }
};
}
}
#endif
