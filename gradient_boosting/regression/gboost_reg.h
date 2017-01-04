#ifndef GBOOST_REG_H_
#define GBOOST_REG_H_
#include "regression/gboost_regdata.h" // DMatrix

namespace gboost {
namespace regression {
class RegBoostLearner {
 public:
  inline void set_param(const char *name, const char *val) {}
  inline void set_data(const DMatrix *train,
                      std::vector<const DMatrix *> evals,
                      std::vector<std::string> evname) {
  }
};
}
}
#endif
