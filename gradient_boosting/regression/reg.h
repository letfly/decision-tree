#ifndef REG_H_
#define REG_H_
#include "booster/gbm.h" // GBMModel, set_param, init_trainer
#include "regression/data.h" // DMatrix, size
#include "utils/stream.h" // IStream

namespace gboost {
namespace regression {
class RegBoostLearner {
 private:
  struct ModelParam {
    float base_score;
    int loss_type;
    inline void set_param(const char *name, const char *val) {
      if (!strcmp("base_score", name)) base_score = (float)atof(val);
      if (!strcmp("loss_type", name)) loss_type = atoi(val);
    }
  };
  ModelParam mparam;
  const DMatrix *train_;
  std::vector<DMatrix *> evals_;
  std::vector<std::string> evname_;
  booster::GBMModel model;
 public:
  inline void set_param(const char *name, const char *val) {
    mparam.set_param(name, val);
    model.set_param(name, val);
  }
  inline void set_data(const DMatrix *train,
                       const std::vector<DMatrix *> &evals,
                       const std::vector<std::string> &evname) {
    this->train_ = train;
    this->evals_ = evals;
    this->evname_ = evname;

    unsigned buffer_size = static_cast<unsigned>(train->size());
    for (size_t i = 0; i < evals.size(); ++i)
      buffer_size += static_cast<unsigned>(evals[i]->size());
    char snum_pbuffer[25];
    printf(snum_pbuffer, "%u", buffer_size);
    model.set_param("num_pbuffer", snum_pbuffer);
  }
  inline void init_trainer(void) { model.init_trainer(); }
  inline void update_one_iter(int iter) {}
  inline void save_model(utils::IStream &fo) const {
    fo.write(&mparam, sizeof(ModelParam));
    //model.save_model(fo);
  }
};
}
}
#endif
