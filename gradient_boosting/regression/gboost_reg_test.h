#ifndef GBOOST_REG_TEST_H_
#define GBOOST_REG_TEST_H_
#include <vector>
#include "regression/gboost_reg.h" // RegBoostLearner
#include "regression/gboost_regdata.h" // DMatrix
#include "utils/gboost_config.h" // ConfigIterator, Assert
#include "utils/gboost_string.h" // StringProcessing

namespace gboost {
namespace regression {
class RegBoostTest {
 private:
  struct TestParam{
    int boost_iterations;
    int save_period;
    char model_dir_path[256];
    char pred_dir_path[256];

    std::vector<std::string> test_paths;
    std::vector<std::string> test_names;

    inline void SetParam(const char *name, const char *val) {
      if (!strcmp("model_dir_path", name)) strcpy(model_dir_path, val);
      if (!strcmp("pred_dir_path", name)) strcpy(pred_dir_path, val);
      if (!strcmp("test_paths", name)) test_paths = utils::StringProcessing::split(val, ';');
      if (!strcmp("test_names", name)) test_names = utils::StringProcessing::split(val, ';');
    }
  };

  TestParam test_param;
  RegBoostLearner *reg_boost_learner;
 public:
  void test(char *config_path, bool silent = false) {
    reg_boost_learner = new RegBoostLearner();
    utils::ConfigIterator config_itr(config_path);
    // Get the training data and validation data paths, config the learner
    while (config_itr.Next()) {
      reg_boost_learner->SetParam(config_itr.name(), config_itr.val());
      test_param.SetParam(config_itr.name(), config_itr.val());
    }

    utils::Assert(test_param.test_paths.size() == test_param.test_names.size(),
      "The number of test data set paths is not the same as the number of test number of test data set data set names");

    // Begin testing
    reg_boost_learner->InitModel();
    char model_path[256];
    std::vector<float> preds;
    for (size_t i = 0; i < test_param.test_paths.size(); ++i) {
      DMatrix test_data;
      test_data.LoadText(test_param.test_paths[i].c_str());
      sprintf(model_path, "%s/final.model", test_param.model_dir_path);
      // Bug: model need to be rb
      utils::FileStream fin(fopen(model_path, "r"));
      reg_boost_learner->LoadModel(fin);
      fin.Close();
      reg_boost_learner->Predict(preds, test_data);
    }
  }
};
}
}
#endif
