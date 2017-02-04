#ifndef LEARNER_OBJECTIVE_H_
#define LEARNER_OBJECTIVE_H_
#include "data.h" // bst_gpair
#include "learner/dmatrix.h" // MetaInfo

namespace gboost {
namespace learner {
// \brief interface of objective function
class IObjFunction{
  public:
  // \brief virtual destructor
  virtual ~IObjFunction(void){}
  // \brief set parameters from outside
  // \param name name of the parameter
  // \param val value of the parameter
  virtual void set_param(const char *name, const char *val) = 0; 
  // \return the default evaluation metric for the objective
  virtual const char* default_eval_metric(void) const = 0;

  // \brief get gradient over each of predictions, given existing information
  // \param preds prediction of current round
  // \param info information about labels, weights, groups in rank
  // \param iter current iteration number
  // \param out_gpair output of get gradient, saves gradient and second order gradient in
  virtual void get_gradient(const std::vector<float> &preds,
                           const MetaInfo &info,
                           int iter,
                           std::vector<bst_gpair> *out_gpair) = 0;
  // the following functions are optional, most of time default implementation is good enough
  // \brief transform prediction values, this is only called when Prediction is called
  // \param io_preds prediction values, saves to this vector as well
  virtual void pred_transform(std::vector<float> *io_preds){}
  // \brief transform prediction values, this is only called when Eval is called, 
  //  usually it redirect to PredTransform
  // \param io_preds prediction values, saves to this vector as well
  virtual void eval_transform(std::vector<float> *io_preds) {
    this->pred_transform(io_preds);
  }
};

// \brief defines functions to calculate some commonly used functions
struct LossType {
  // \brief indicate which type we are using
  int loss_type;
  // list of constants
  static const int kLinearSquare = 0;
  // \brief get default evaluation metric for the objective
  inline const char *default_eval_metric(void) const {
    return "rmse";
  }

  // \brief transform the linear sum to prediction
  // \param x linear sum of boosting ensemble
  // \return transformed prediction
  inline float pred_transform(float x) const {
    switch (loss_type) {
      case kLinearSquare: {printf("ddd");return x;}
      default: utils::error("unknown loss_type"); return 0.0f;
    }
  }
  // \brief calculate first order gradient of loss, given transformed prediction
  // \param predt transformed prediction
  // \param label true label
  // \return first order gradient
  inline float first_order_gradient(float predt, float label) const {
    switch (loss_type) {
      case kLinearSquare: return predt - label;
      default: utils::error("unknown loss_type"); return 0.0f;
    }
  }
  // \brief calculate second order gradient of loss, given transformed prediction
  // \param predt transformed prediction
  // \param label true label
  // \return second order gradient
  inline float second_order_gradient(float predt, float label) const {
    switch (loss_type) {
      case kLinearSquare: return 1.0f;
      default: utils::error("unknown loss_type"); return 0.0f;
    }
  }
};
class RegLossObj : public IObjFunction{
 private:
  LossType loss;
  float scale_pos_weight;
 public:
  explicit RegLossObj(int loss_type) {
    loss.loss_type = loss_type;
    scale_pos_weight = 1.0f;
  }
  virtual ~RegLossObj(void) {}
  virtual void set_param(const char *name, const char *val) {
    if (!strcmp("scale_pos_weight", name))
      scale_pos_weight = static_cast<float>(atof(val));
  }
  virtual const char* default_eval_metric(void) const {
    return loss.default_eval_metric();
  }

  virtual void get_gradient(const std::vector<float> &preds,
                           const MetaInfo &info,
                           int iter,
                           std::vector<bst_gpair> *out_gpair) {
    utils::check(info.labels.size() != 0, "label set cannot be empty");
    utils::check(preds.size() % info.labels.size() == 0,
                 "labels are not correctly provided");
    std::vector<bst_gpair> &gpair = *out_gpair;
    gpair.resize(preds.size());
    // start calculating gradient
    const unsigned nstep = static_cast<unsigned>(info.labels.size());
    const bst_uint ndata = static_cast<bst_uint>(preds.size());
    for (bst_uint i = 0; i < ndata; ++i) {
      const unsigned j = i % nstep;
      float p = loss.pred_transform(preds[i]);
      float w = info.get_weight(j);
      if (info.labels[j] == 1.0f) w *= scale_pos_weight;
      gpair[i] = bst_gpair(loss.first_order_gradient(p, info.labels[j]) * w,
                           loss.second_order_gradient(p, info.labels[j]) * w);
    }
  }
  virtual void pred_transform(std::vector<float> *io_preds) {
    std::vector<float> &preds = *io_preds;
    const bst_uint ndata = static_cast<bst_uint>(preds.size());
    for (bst_uint j = 0; j < ndata; ++j) {
      preds[j] = loss.pred_transform(preds[j]);
    }
  }
};
// this are implementations of objective functions
// factory function
// \brief factory funciton to create objective function by name
inline IObjFunction* CreateObjFunction(const char *name) {
  if (!strcmp("reg:linear", name)) return new RegLossObj(LossType::kLinearSquare);
  utils::error("unknown objective function type: %s", name);
  return NULL;
}

}
}
#endif
