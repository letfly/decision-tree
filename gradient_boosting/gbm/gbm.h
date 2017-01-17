#ifndef GBM_GBM_H_
#define GBM_GBM_H_
#include "tree/updater.h" //IUpdater
#include "tree/model.h" // RegTree

namespace gboost {
namespace gbm {
class IGradBooster {
 public:
  // \brief set parameters from outside
  // \param name name of the parameter
  // \param val  value of the parameter
  virtual void set_param(const char *name, const char *val) = 0;
  // \brief initialize the model
  virtual void init_model(void) = 0;
  // destrcutor
  virtual ~IGradBooster(void){}
};
class GBTree : public IGradBooster {
 private:
  // configurations for tree
  std::vector< std::pair<std::string, std::string> > cfg;
  // the updaters that can be applied to each of tree
  std::vector<tree::IUpdater*> updaters;

  // --- data structure ---
  // \brief training parameters
  struct TrainParam {
    // \brief tree updater sequence
    std::string updater_seq;
    // \brief whether updater is already initialized
    int updater_initialized;
    // construction
    TrainParam(void) {
      updater_seq = "grow_colmaker,prune";
      updater_initialized = 0;
    }
    inline void set_param(const char *name, const char *val){
      if (!strcmp(name, "updater") &&
          strcmp(updater_seq.c_str(), val) != 0) {
        updater_seq = val;
        updater_initialized = 0;
      }
    }
  };
  // training parameter
  TrainParam tparam;

  // \brief vector of trees stored in the model
  std::vector<tree::RegTree*> trees;

  // \brief model parameters
  struct ModelParam {
    // \brief number of root: default 0, means single tree
    int num_roots;
    // \brief number of features to be used by trees
    int num_feature;
    // \brief size of predicton buffer allocated used for buffering
    int64_t num_pbuffer;

    // \brief how many output group a single instance can produce
    //  this affects the behavior of number of output we have:
    //    suppose we have n instance and k group, output will be k*n 
    int num_output_group;
    // \brief size of leaf vector needed in tree
    int size_leaf_vector;
    // \brief number of trees
    int num_trees;
    // \brief constructor
    ModelParam(void) {
      num_roots = num_feature = 0;
      num_pbuffer = 0;
      num_output_group = 1;
      size_leaf_vector = 0;
      num_trees = 0;
    }
    // \brief set parameters from outside
    // \param name name of the parameter
    // \param val  value of the parameter
    inline void set_param(const char *name, const char *val) {
      using namespace std;
      if (!strcmp("bst:num_roots", name)) num_roots = atoi(val);
      if (!strcmp("bst:num_feature", name)) num_feature = atoi(val);
      if (!strcmp("num_pbuffer", name)) num_pbuffer = atol(val);
      if (!strcmp("num_output_group", name)) num_output_group = atol(val);
      if (!strcmp("bst:size_leaf_vector", name)) size_leaf_vector = atoi(val);
    }
    // \return size of prediction buffer actually needed
    inline size_t pred_buffer_size(void) const {
      return num_output_group * num_pbuffer * (size_leaf_vector + 1);
    }
  };
  // model parameter
  ModelParam mparam;

  // \brief prediction buffer
  std::vector<float>  pred_buffer;
  // \brief prediction buffer counter, remember the prediction
  std::vector<unsigned> pred_counter;
 public:
  virtual void set_param(const char *name, const char *val) {
    if (!strncmp(name, "bst:", 4)) {
      cfg.push_back(std::make_pair(std::string(name+4), std::string(val)));
      // set into updaters, if already intialized
      for (size_t i = 0; i < updaters.size(); ++i)
        updaters[i]->set_param(name+4, val);
    }
    if (!strcmp(name, "silent")) this->set_param("bst:silent", val);
    tparam.set_param(name, val);
    if (trees.size() == 0) mparam.set_param(name, val);
  }
  // initialize the predic buffer
  virtual void init_model(void) {
    pred_buffer.clear(); pred_counter.clear();
    pred_buffer.resize(mparam.pred_buffer_size(), 0.0f);
    pred_counter.resize(mparam.pred_buffer_size(), 0);
    utils::assert(mparam.num_trees == 0, "GBTree: model already initialized");
    utils::assert(trees.size() == 0, "GBTree: model already initialized");
  }
};
class GBLinear : public IGradBooster {
 private:
  // training parameter
  struct ParamTrain {
    // \brief learning_rate
    float learning_rate;
    // \brief regularization weight for L2 norm
    float reg_lambda;
    // \brief regularization weight for L1 norm
    float reg_alpha;
    // \brief regularization weight for L2 norm in bias
    float reg_lambda_bias;
    // parameter
    ParamTrain(void) {
      reg_alpha = 0.0f;
      reg_lambda = 0.0f;
      reg_lambda_bias = 0.0f;
      learning_rate = 1.0f;
    }
    inline void set_param(const char *name, const char *val) {
      // sync-names
      if (!strcmp("eta", name)) learning_rate = static_cast<float>(atof(val));
      if (!strcmp("lambda", name)) reg_lambda = static_cast<float>(atof(val));
      if (!strcmp( "alpha", name)) reg_alpha = static_cast<float>(atof(val));
      if (!strcmp( "lambda_bias", name)) reg_lambda_bias = static_cast<float>(atof(val));
      // real names
      if (!strcmp( "learning_rate", name)) learning_rate = static_cast<float>(atof(val));
      if (!strcmp( "reg_lambda", name)) reg_lambda = static_cast<float>(atof(val));
      if (!strcmp( "reg_alpha", name)) reg_alpha = static_cast<float>(atof(val));
      if (!strcmp( "reg_lambda_bias", name)) reg_lambda_bias = static_cast<float>(atof(val));
    }
  };
  // training parameter
  ParamTrain param;

  // model for linear booster
  class Model {
   public:
    // model parameter
    struct Param {
      // number of feature dimension
      int num_feature;
      // number of output group
      int num_output_group;
      // constructor
      Param(void) {
        num_feature = 0;
        num_output_group = 1;
      }
      inline void set_param(const char *name, const char *val) {
        if (!strcmp(name, "bst:num_feature")) num_feature = atoi(val);
        if (!strcmp(name, "num_output_group")) num_output_group = atoi(val);
      }
    };
    // parameter
    Param param;
    // weight for each of feature, bias is the last one
    std::vector<float> weight;
    // initialize the model parameter
    inline void init_model(void) {
      // bias is the last weight
      weight.resize((param.num_feature + 1) * param.num_output_group);
      std::fill(weight.begin(), weight.end(), 0.0f);
    }
  };
  // model field
  Model model;
 public:
  // set model parameters
  virtual void set_param(const char *name, const char *val) {
    if (!strncmp(name, "bst:", 4)) param.set_param(name + 4, val);
    if (model.weight.size() == 0) model.param.set_param(name, val);
  }
  virtual void init_model(void) {
    model.init_model();
  }
};

IGradBooster* create_grad_booster(const char *name) {
  if (!strcmp("gbtree", name)) return new GBTree();
  if (!strcmp("gblinear", name)) return new GBLinear();
  utils::error("unknown booster type: %s", name);
  return NULL;
}

}
}
#endif
