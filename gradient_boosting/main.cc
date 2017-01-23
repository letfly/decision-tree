#include <vector>
#include "io/io.h"
#include "learner/dmatrix.h"
#include "learner/learner.h" // BoostLearner
#include "utils/config.h" // ConfigIterator, assert
#include "utils/utils.h" // fopen_check

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
      gboost::utils::assert(sscanf(name, "eval[%[^]]", evname)==1, "must specify evaluation name for display");
      eval_data_names.push_back(std::string(evname));
      eval_data_paths.push_back(std::string(val));
    }
    learner.set_param(name, val);
  }

  gboost::learner::DMatrix *train_data;
  int silent; // Whether use printf
  int use_buffer; // Whether use binary buffer
  std::vector<gboost::learner::DMatrix *> test_data_vec;
  std::vector<const gboost::learner::DMatrix *> test_data_vecall;
  int eval_train; // Whether evaluate train sets
  inline void init_data() {
    train_data = gboost::io::load_data_matrix(train_path.c_str(), silent!=0, use_buffer!=0);
    gboost::utils::assert(eval_data_names.size() == eval_data_paths.size(), "BUG");
    for (size_t i = 0; i < eval_data_paths.size(); ++i) {
      test_data_vec.push_back(gboost::io::load_data_matrix(eval_data_paths[i].c_str(), silent!=0, use_buffer!=0));
      test_data_vecall.push_back(test_data_vec.back());
    }

    std::vector<gboost::learner::DMatrix *> dcache(1, train_data);
    for (size_t i = 0; i < test_data_vec.size(); ++i) dcache.push_back(test_data_vec[i]);
    // Set cache data to be all train and evaluation data
    learner.set_cache_data(dcache);

    // Add train set to evaluation set if needed
    if (eval_train != 0) {
      test_data_vecall.push_back(train_data);
      eval_data_names.push_back(std::string("train"));
    }
  }

  inline void init_learner() {
    learner.init_model();
  }

  gboost::learner::DMatrix *data;
  inline void train() {
    const time_t start = time(NULL);
    unsigned long elapsed = 0;
    learner.check_init(data);
    for (int i = 0; i < num_round; ++i) {
      elapsed = (unsigned long)(time(NULL) - start);
      if (!silent) printf("boosting round %d, %lu sec elapsed\n", i, elapsed);
      learner.update_one_iter(i, *data);
      std::string res = learner.eval_one_iter(i, test_data_vecall, eval_data_names);
      fprintf(stderr, "%s\n", res.c_str());
      elapsed = (unsigned long)(time(NULL) - start);
    }
    if (!silent) printf("\nupdating end, %lu sec in all\n", elapsed);
  }
 public:
  ~Task(void){
    for (size_t i = 0; i < test_data_vec.size(); i++){
      delete test_data_vec[i];
    }
    if (train_data != NULL) delete train_data;
  }
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
