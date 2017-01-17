#ifndef LEARNER_OBJECTIVE_H_
#define LEARNER_OBJECTIVE_H_

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
