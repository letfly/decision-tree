#ifndef REG_TRAIN_H_
#define REG_TRAIN_H_
#include <string>
#include <vector>
#include "regression/reg.h" // RegBoostLearner, set_param, set_data, init_trainer, update_one_iter, save_model
#include "regression/data.h" // DMatrix, load_text
#include "utils/config.h" // ConfigIterator, assert, next, name, val
#include "utils/string.h" // StringProcessing, split

namespace gboost {
namespace regression {
class RegBoostTrain {
 private:
  RegBoostLearner *reg_boost_learner;
  struct TrainParam {
    int boost_iterations;
    int save_period;
    char train_path[256];
    char model_dir_path[256];

    std::vector<std::string> validation_data_paths;
    std::vector<std::string> validation_data_names;

    inline void set_param(const char *name, const char *val) {
      if (!strcmp("boost_iterations", name)) boost_iterations = atoi(val);
      if (!strcmp("save_period", name)) save_period = atoi(val);
      if (!strcmp("train_path", name)) strcpy(train_path, val);
      if (!strcmp("model_dir_path", name)) strcpy(model_dir_path, val);
      if (!strcmp("validation_paths", name)) {
        validation_data_paths = utils::StringProcessing::split(val, ';');
      }
      if (!strcmp("validation_names", name)) {
        validation_data_names = utils::StringProcessing::split(val, ';');
      }
    }
  };
  TrainParam train_param;
  void save_model(const char* suffix) {
    char model_path[256];
    sprintf(model_path, "%s/%s", train_param.model_dir_path, suffix);
    FILE* file = fopen(model_path, "w");
    utils::FileStream fin(file);
    reg_boost_learner->save_model(fin);
    fin.close();
  }
 public:
  void train(char *config_path, bool silent = false) {
    // Init the path
    reg_boost_learner = new RegBoostLearner();
    // Get the training data and validation data paths
    utils::ConfigIterator config_itr(config_path);
    // Config the learner
    while (config_itr.next()) {
      reg_boost_learner->set_param(config_itr.name(), config_itr.val());
      train_param.set_param(config_itr.name(), config_itr.val());
    }
    utils::assert(train_param.validation_data_paths.size() == train_param.validation_data_names.size(),
      "The number of validation paths is not the same as the number of validation data set names");

    // Input
    DMatrix train;
    train.load_text(train_param.train_path, silent);
    std::vector<DMatrix *> evals;
    for (int i = 0; i < train_param.validation_data_paths.size(); ++i) {
      DMatrix eval;
      eval.load_text(train_param.validation_data_paths[i].c_str(), silent);
      evals.push_back(&eval);
    }
    reg_boost_learner->set_data(&train, evals, train_param.validation_data_names);

    // Model build
    reg_boost_learner->init_trainer();
    char suffix[256];
    for (int i = 1; i <= train_param.boost_iterations; ++i) {
      reg_boost_learner->update_one_iter(i);
      if (train_param.save_period!=0 && i%train_param.save_period==0) {
        sprintf(suffix, "%d.model", i);
        printf("ddddd");
        save_model(suffix);
      }
    }
    save_model("final.model");
  }
};
}
}
#endif
