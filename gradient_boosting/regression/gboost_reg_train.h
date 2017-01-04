#ifndef GBOOST_REG_TRAIN_H_
#define GBOOST_REG_TRAIN_H_
#include <string>
#include <vector>
#include "regression/gboost_reg.h" // RegBoostLearner, set_param, set_data
#include "regression/gboost_regdata.h" // DMatrix, load_text
#include "utils/gboost_config.h" // ConfigIterator, assert, next, name, val
#include "utils/gboost_string.h" // StringProcessing, split

namespace gboost {
namespace regression {
class RegBoostTrain {
 private:
  RegBoostLearner *reg_boost_learner;
  struct TrainParam {
    int boost_iterations;
    int save_period;
    char train_path[256];
    const char *model_dir_path;

    std::vector<std::string> validation_data_paths;
    std::vector<std::string> validation_data_names;

    inline void set_param(const char *name, const char *val) {
      if (!strcmp("boost_iterations", name)) boost_iterations = atoi(val);
      if (!strcmp("save_period", name)) save_period = atoi(val);
      if (!strcmp("train_path", name)) strcpy(train_path, val);
      if (!strcmp("model_dir_path", name)) model_dir_path = val;
      if (!strcmp("validation_paths", name)) {
        validation_data_paths = utils::StringProcessing::split(val, ';');
      }
      if (!strcmp("validation_names", name)) {
        validation_data_names = utils::StringProcessing::split(val, ';');
      }
    }
  };
  TrainParam train_param;
 public:
  void train(char *config_path, bool silent = false) {
    // Init the path
    reg_boost_learner = new RegBoostLearner();
    // Get the training data and validation data paths
    utils::ConfigIterator config_itr(config_path);
    // Config the learner
    while (config_itr.next()) {
      printf("name=%sval=%s\n", config_itr.name(), config_itr.val());
      reg_boost_learner->set_param(config_itr.name(), config_itr.val());
      train_param.set_param(config_itr.name(), config_itr.val());
    }
    utils::assert(train_param.validation_data_paths.size() == train_param.validation_data_names.size(),
      "The number of validation paths is not the same as the number of validation data set names");

    // Input
    DMatrix train;
    //train.load_text(train_param.train_path, silent);
    std::vector<const DMatrix *> evals;
    for (int i = 0; i < train_param.validation_data_paths.size(); ++i) {
      DMatrix eval;
      eval.load_text(train_param.validation_data_paths[i].c_str(), silent);
      evals.push_back(&eval);
    }
    reg_boost_learner->set_data(&train, evals, train_param.validation_data_names);
  }
};
}
}
#endif
