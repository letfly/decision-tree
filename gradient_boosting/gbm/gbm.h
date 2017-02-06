#ifndef GBM_GBM_H_
#define GBM_GBM_H_
#include "data.h" // RowBatch, bst_gpair
#include "tree/updater.h" //IUpdater
#include "tree/model.h" // RegTree
#include "utils/iterator.h" // IIterator

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

  // \brief generate predictions for given feature matrix
  // \param p_fmat feature matrix
  // \param buffer_offset buffer index offset of these instances, if equals -1
  //        this means we do not have buffer index allocated to the gbm
  //  a buffer index is assigned to each instance that requires repeative prediction
  //  the size of buffer is set by convention using IGradBooster.SetParam("num_pbuffer","size")
  // \param info extra side information that may be needed for prediction
  // \param out_preds output vector to hold the predictions
  // \param ntree_limit limit the number of trees used in prediction, when it equals 0, this means 
  //    we do not limit number of trees, this parameter is only valid for gbtree, but not for gblinear
  virtual void predict(IFMatrix *p_fmat,
                       int64_t buffer_offset,
                       const BoosterInfo &info,
                       std::vector<float> *out_preds,
                       unsigned ntree_limit = 0) = 0;

  // \brief peform update to the model(boosting)
  // \param p_fmat feature matrix that provide access to features
  // \param info meta information about training
  // \param in_gpair address of the gradient pair statistics of the data
  // the booster may change content of gpair
  virtual void do_boost(IFMatrix *p_fmat,
                       const BoosterInfo &info,
                       std::vector<bst_gpair> *in_gpair) = 0;
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
    // \brief get the buffer offset given a buffer index and group id  
    // \return calculated buffer offset
    inline int64_t buffer_offset(int64_t buffer_index, int bst_group) const {
      if (buffer_index < 0) return -1;
      utils::check(buffer_index < num_pbuffer, "buffer_index exceed num_pbuffer");
      return (buffer_index + num_pbuffer * bst_group) * (size_leaf_vector + 1);
    }
  };
  // model parameter
  ModelParam mparam;

  // \brief prediction buffer
  std::vector<float>  pred_buffer;
  // \brief prediction buffer counter, remember the prediction
  std::vector<unsigned> pred_counter;

  // temporal storage for per thread
  std::vector<tree::RegTree::FVec> thread_temp;
  // \brief some information indicator of the tree, reserved
  std::vector<int> tree_info;
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
  // make a prediction for a single instance
  inline void pred(const RowBatch::Inst &inst,
                   int64_t buffer_index,
                   int bst_group,
                   unsigned root_index,
                   tree::RegTree::FVec *p_feats,
                   float *out_pred, size_t stride, unsigned ntree_limit) {
    size_t itop = 0;
    float  psum = 0.0f;
    // sum of leaf vector 
    std::vector<float> vec_psum(mparam.size_leaf_vector, 0.0f);
    const int64_t bid = mparam.buffer_offset(buffer_index, bst_group);
    // number of valid trees
    unsigned treeleft = ntree_limit == 0 ? std::numeric_limits<unsigned>::max() : ntree_limit;
    // load buffered results if any
    if (bid >= 0 && ntree_limit == 0) {
      itop = pred_counter[bid];
      psum = pred_buffer[bid];
      for (int i = 0; i < mparam.size_leaf_vector; ++i) {
        vec_psum[i] = pred_buffer[bid + i + 1];
      }
    }
    if (itop != trees.size()) {
      p_feats->fill(inst);
      for (size_t i = itop; i < trees.size(); ++i) {
        if (tree_info[i] == bst_group) {
          int tid = trees[i]->get_leaf_index(*p_feats, root_index);
          psum += (*trees[i])[tid].leaf_value();
          for (int j = 0; j < mparam.size_leaf_vector; ++j) {
            vec_psum[j] += trees[i]->leafvec(tid)[j];
          }
          if(--treeleft == 0) break;
        }
      }
      p_feats->drop(inst);
    }
    // updated the buffered results
    if (bid >= 0 && ntree_limit == 0) {
      pred_counter[bid] = static_cast<unsigned>(trees.size());
      pred_buffer[bid] = psum;
      for (int i = 0; i < mparam.size_leaf_vector; ++i) {
        pred_buffer[bid + i + 1] = vec_psum[i];
      }
    }
    out_pred[0] = psum;
    for (int i = 0; i < mparam.size_leaf_vector; ++i) {
      out_pred[stride * (i + 1)] = vec_psum[i];
    }
  }
  virtual void predict(IFMatrix *p_fmat,
                       int64_t buffer_offset,
                       const BoosterInfo &info,
                       std::vector<float> *out_preds,
                       unsigned ntree_limit = 0) {
    int nthread = 1;
    thread_temp.resize(nthread, tree::RegTree::FVec());
    for (int i = 0; i < nthread; ++i) {
      thread_temp[i].init(mparam.num_feature);
    }

    std::vector<float> &preds = *out_preds;
    const size_t stride = info.num_row * mparam.num_output_group;
    preds.resize(stride * (mparam.size_leaf_vector+1));
    // start collecting the prediction
    utils::IIterator<RowBatch> *iter = p_fmat->row_iterator();
    iter->before_first();
    while (iter->next()) {
      const RowBatch &batch = iter->value();
      // parallel over local batch
      const bst_uint nsize = static_cast<bst_uint>(batch.size);
      #pragma omp parallel for schedule(static)
      for (bst_uint i = 0; i < nsize; ++i) {
        const int tid = 0;
        tree::RegTree::FVec &feats = thread_temp[tid];
        int64_t ridx = static_cast<int64_t>(batch.base_rowid + i);
        utils::assert(static_cast<size_t>(ridx) < info.num_row, "data row index exceed bound");
        // loop over output groups
        for (int gid = 0; gid < mparam.num_output_group; ++gid) {
          this->pred(batch[i],
                     buffer_offset < 0 ? -1 : buffer_offset + ridx,
                     gid, info.get_root(ridx), &feats,
                     &preds[ridx * mparam.num_output_group + gid], stride, 
                     ntree_limit);
        }
      }
    }
  }
  virtual void do_boost(IFMatrix *p_fmat,
                        const BoosterInfo &info,
                        std::vector<bst_gpair> *in_gpair) {
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

    // given original weight calculate delta bias
    inline double calc_delta_bias(double sum_grad, double sum_hess, double w) {
      return - (sum_grad + reg_lambda_bias * w) / (sum_hess + reg_lambda_bias);
    }
    // given original weight calculate delta
    inline double calc_delta(double sum_grad, double sum_hess, double w) {
      if (sum_hess < 1e-5f) return 0.0f;
      double tmp = w - (sum_grad + reg_lambda * w) / (sum_hess + reg_lambda);
      if (tmp >=0) {
        return std::max(-(sum_grad + reg_lambda * w + reg_alpha) / (sum_hess + reg_lambda), -w);
      } else {
        return std::min(-(sum_grad + reg_lambda * w - reg_alpha) / (sum_hess + reg_lambda), -w);
      }
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
        if (!strcmp(name, "bst:num_feature")) {printf("nnn%s",val);num_feature = atoi(val);}
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

    // get i-th weight
    inline float* operator[](size_t i) {
      printf("i=%d,%f", i,weight[i*param.num_output_group]);
      return &weight[i * param.num_output_group];
    }
    // model bias
    inline float* bias(void) {
      return &weight[param.num_feature * param.num_output_group];
    }
  };
  // model field
  Model model;

  inline void pred(const RowBatch::Inst &inst, float *preds) {
    for (int gid = 0; gid < model.param.num_output_group; ++gid) {
      float psum = model.bias()[gid];
      printf("pred=%f", psum);
      for (bst_uint i = 0; i < inst.length; ++i) {
        psum += inst[i].fvalue * model[inst[i].index][gid];
      }
      preds[gid] = psum;
    }
  }
 public:
  // set model parameters
  virtual void set_param(const char *name, const char *val) {
    if (!strncmp(name, "bst:", 4)) param.set_param(name + 4, val);
    if (model.weight.size() == 0) model.param.set_param(name, val);
  }
  virtual void init_model(void) {
    model.init_model();
  }
  virtual void predict(IFMatrix *p_fmat,
                       int64_t buffer_offset,
                       const BoosterInfo &info,
                       std::vector<float> *out_preds,
                       unsigned ntree_limit = 0) {
    utils::check(ntree_limit == 0,
                 "GBLinear::predict ntrees is only valid for gbtree predictor");
    std::vector<float> &preds = *out_preds;
    preds.resize(0);
    // start collecting the prediction
    utils::IIterator<RowBatch> *iter = p_fmat->row_iterator();
    const int ngroup = model.param.num_output_group;
    while (iter->next()) {
      const RowBatch &batch = iter->value();
      utils::assert(batch.base_rowid * ngroup == preds.size(),
                    "base_rowid is not set correctly");
      // output convention: nrow * k, where nrow is number of rows
      // k is number of group
      preds.resize(preds.size() + batch.size * ngroup);
      // parallel over local batch
      const bst_uint nsize = static_cast<bst_uint>(batch.size);
      for (bst_uint i = 0; i < nsize; ++i) {
        const size_t ridx = batch.base_rowid + i;
        // loop over output groups
        for (int gid = 0; gid < ngroup; ++gid) {
          printf("predictpred");
          this->pred(batch[i], &preds[ridx * ngroup]);
        }
      }
      for (auto i: preds) printf("predict=%f", i);
    }
  }

  virtual void do_boost(IFMatrix *p_fmat,
                       const BoosterInfo &info,
                       std::vector<bst_gpair> *in_gpair) {
    std::vector<bst_gpair> &gpair = *in_gpair;
    const int ngroup = model.param.num_output_group;
    const std::vector<bst_uint> &rowset = p_fmat->buffered_rowset();
    // for all the output group
    for (int gid = 0; gid < ngroup; ++gid) {
      double sum_grad = 0.0, sum_hess = 0.0;
      const bst_uint ndata = static_cast<bst_uint>(rowset.size());
      for (bst_uint i = 0; i < ndata; ++i) {
        bst_gpair &p = gpair[rowset[i] * ngroup + gid];
        if (p.hess >= 0.0f) {
          sum_grad += p.grad; sum_hess += p.hess;
        }
      }
      // remove bias effect
      bst_float dw = static_cast<bst_float>(
          param.learning_rate * param.calc_delta_bias(sum_grad, sum_hess, model.bias()[gid]));
      model.bias()[gid] += dw;
      printf("dw=%f", dw);
      // update grad value
      for (bst_uint i = 0; i < ndata; ++i) {
        bst_gpair &p = gpair[rowset[i] * ngroup + gid];
        if (p.hess >= 0.0f) {
          p.grad += p.hess * dw;
        }
      }
    }
    utils::IIterator<ColBatch> *iter = p_fmat->col_iterator();
    while (iter->next()) {
      // number of features
      const ColBatch &batch = iter->value();
      const bst_uint nfeat = static_cast<bst_uint>(batch.size);
      for (bst_uint i = 0; i < nfeat; ++i) {
        const bst_uint fid = batch.col_index[i];
        ColBatch::Inst col = batch[i];
        for (int gid = 0; gid < ngroup; ++gid) {
          double sum_grad = 0.0, sum_hess = 0.0;
          for (bst_uint j = 0; j < col.length; ++j) {
            const float v = col[j].fvalue;
            bst_gpair &p = gpair[col[j].index * ngroup + gid];
            if (p.hess < 0.0f) continue;
            sum_grad += p.grad * v;
            sum_hess += p.hess * v * v;
          }
          float &w = model[fid][gid];
          bst_float dw = static_cast<bst_float>(param.learning_rate * param.calc_delta(sum_grad, sum_hess, w));
          w += dw;
          // update grad value
          for (bst_uint j = 0; j < col.length; ++j) {
            bst_gpair &p = gpair[col[j].index * ngroup + gid];
            if (p.hess < 0.0f) continue;
            p.grad += p.hess * col[j].fvalue * dw;
          }
        }
      }
    }
  }
};

IGradBooster* create_grad_booster(const char *name) {
  //if (!strcmp("gbtree", name)) return new GBTree();
  if (!strcmp("gblinear", name)) return new GBLinear();
  if (!strcmp("gbtree", name)) return new GBTree();
  utils::error("unknown booster type: %s", name);
  return NULL;
}

}
}
#endif
