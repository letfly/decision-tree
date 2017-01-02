#ifndef GBOOST_REG_TRAIN_H_
#define GBOOST_REG_TRAIN_H_
#include <string>
#include <vector>
#include "regression/gboost_reg.h" // RegBoostLearner
#include "regression/gboost_regdata.h" // DMatrix
#include "utils/gboost_config.h" // ConfigIterator, Assert
#include "utils/gboost_string.h" // StringProcessing

namespace gboost {
namespace regression {
class RegBoostTrain {
 private:
  struct TrainParam {
    int boost_iterations;
    int save_period;
    const char *train_path;
    const char *model_dir_path;

    std::vector<std::string> validation_data_paths;
    std::vector<std::string> validation_data_names;

    inline void SetParam(const char *name, const char *val) {
      if (!strcmp("boost_iterations", name)) boost_iterations = atoi(val);
      if (!strcmp("save_period", name)) save_period = atoi(val);
      if (!strcmp("train_path", name)) train_path = val;
      if (!strcmp("model_dir_path", name)) model_dir_path = val;
      if (!strcmp("validation_paths", name)) {
        validation_data_paths = utils::StringProcessing::split(val, ';');
      }
      if (!strcmp("validation_names", name)) {
        validation_data_names = utils::StringProcessing::split(val, ';');
      }
    }
  };

  RegBoostLearner *reg_boost_learner;
  TrainParam train_param;
 public:
  void train(char *config_path, bool silent = false) {
    reg_boost_learner = new RegBoostLearner();
    utils::ConfigIterator config_itr(config_path);
    // Get the training data and validation data paths, config the Learner
    while (config_itr.Next()) {
      reg_boost_learner->SetParam(config_itr.name(), config_itr.val());
      train_param.SetParam(config_itr.name(), config_itr.val());
    }

    utils::Assert(train_param.validation_data_paths.size() == train_param.validation_data_names.size(),
      "The number of validation paths is not the same as the number of validation data set names");

    // Load Data
    DMatrix train;
    train.LoadText(train_param.train_path);
    std::vector<const DMatrix *> evals;
    for (int i = 0; i < train_param.validation_data_paths.size(); ++i) {
      DMatrix eval;
      eval.LoadText(train_param.validation_data_paths[i].c_str(), silent);
      evals.push_back(&eval);
    }
    reg_boost_learner->SetData(&train, evals, train_param.validation_data_names);

    // Begin training
    reg_boost_learner->InitTrainer();
    char model_path[256];
    for (int i = 1; i <= train_param.boost_iterations; ++i) {
      reg_boost_learner->UpdateOneIter(i);
      // Save the models during the iterations
      if (train_param.save_period!=0 && i%train_param.save_period==0) {
        sscanf(model_path, "%s/%d.model", train_param.model_dir_path, i);
        FILE *file = fopen(model_path, "w");
        utils::FileStream fin(file);
        reg_boost_learner->SaveModel(fin);
        fin.Close();
      }
    }

    // Save the final model
    sscanf(model_path, "%s/final.model", train_param.model_dir_path);
    FILE *file = fopen(model_path, "w");
    utils::FileStream fin(file);
    reg_boost_learner->SaveModel(fin);
    fin.Close();
  }
};
}
}
#endif
