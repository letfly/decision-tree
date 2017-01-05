#include <vector>
#include "learner/learner.h"
#include "utils/config.h"

class Task {
 private:
  int num_round; // Number of boosting iterations
  int save_period; // The period to save the model
  std::string train_path, test_path; // The path of train/test data set
  std::vector<std::string> eval_data_names; // The names of evaluation data sets
  std::vector<std::string> eval_data_paths; // The paths of validation data sets
  gboost::learner::BoostLearner learner;
  inline void set_param(const char *name, const char *val) {
    if (!strcmp("num_round", name)) num_round = atoi(val);
    if (!strcmp("save_period", name)) save_period = atoi(val);
    if (!strcmp("train_path", name)) train_path = val;
    if (!strcmp("test_path", name)) test_path = val;
    if (!strncmp("eval[", name, 5)) {
      char evname[256];
      gboost::utils::assert(sscanf(name, "eval{%[^]]", evname)==1, "must specify evaluation name for display");
      eval_data_names.push_back(std::string(evname));
      eval_data_paths.push_back(std::string(val));
    }
    learner.set_param(name, val);
  }

  inline void init_data() {}

  inline void init_learner() {}

  inline void train() {}
 public:
  inline void run(int argc, char *argv[]) {
    if (argc < 2) {
      printf("please use use config file\n");
      return;
    }
    gboost::utils::ConfigIterator itr(argv[1]);
    while (itr.next()) this->set_param(itr.name(), itr.val());
    for (int i = 2; i < argc; ++i) {
      char name[256], val[256];
      if (sscanf(argv[i], "%[^=]=%s", name, val) == 2) this->set_param(name, val);
    }
    this->init_data();
    this->init_learner();
    this->train();
  }
};

int main(int argc, char *argv[]) {
  Task tsk;
  tsk.run(argc, argv);

  return 0;
}
