#ifndef LEARNER_EVALUATION_H_
#define LEARNER_EVALUATION_H_

namespace gboost {
namespace learner {
// \brief evaluator that evaluates the loss metrics
struct IEvaluator{
  // \return name of metric
  virtual const char *name(void) const = 0;
  // \brief virtual destructor
  virtual ~IEvaluator(void) {}
};

struct EvalEWiseBase : public IEvaluator {
};
// \brief RMSE
struct EvalRMSE : public EvalEWiseBase {
  virtual const char *name(void) const { return "rmse"; }
};
// \brief error
struct EvalError : public EvalEWiseBase {
  virtual const char *name(void) const { return "error"; }
};
// \brief match error
struct EvalMatchError : public EvalEWiseBase {
  virtual const char *name(void) const { return "merror"; }
};
// \brief logloss
struct EvalLogLoss : public EvalEWiseBase {
  virtual const char *name(void) const { return "logloss"; }
};

// \brief Area under curve, for both classification and rank
class EvalAuc : public IEvaluator {
  virtual const char *name(void) const { return "auc"; }
};
// \brief AMS: also records best threshold
class EvalAMS : public IEvaluator {
 private:
  std::string name_;
  float ratio_;
 public:
  explicit EvalAMS(const char *name) {
    name_ = name;
    // note: ams@0 will automatically select which ratio to go
    utils::check(std::sscanf(name, "ams@%f", &ratio_) == 1, "invalid ams format");
  }
  virtual const char *name(void) const { return name_.c_str(); }
};

// \brief Evaluate rank list
class EvalRankList : public IEvaluator {
 private:
  std::string name_;
  bool minus_;
  unsigned topn_;
 protected:
  explicit EvalRankList(const char *name) {
    name_ = name;
    minus_ = false;
    if (sscanf(name, "%*[^@]@%u[-]?", &topn_) != 1) topn_ = UINT_MAX;
    if (name[strlen(name) - 1] == '-') minus_ = true;
  }
 public:
  virtual const char *name(void) const { return name_.c_str(); }
};
// \brief Precison at N, for both classification and rank
class EvalPrecision : public EvalRankList{
 public:
  explicit EvalPrecision(const char *name) : EvalRankList(name) {}
};
// \brief Precison at N, for both classification and rank
class EvalMAP : public EvalRankList {
 public:
  explicit EvalMAP(const char *name) : EvalRankList(name) {}
};
// \brief NDCG
class EvalNDCG : public EvalRankList{
 public:
  explicit EvalNDCG(const char *name) : EvalRankList(name) {}
};

// \brief precision with cut off at top percentile
class EvalPrecisionRatio : public IEvaluator{
 private:
  int use_ap;
  float ratio_;
  std::string name_;
 public:
  explicit EvalPrecisionRatio(const char *name) : name_(name) {
    if (sscanf(name, "apratio@%f", &ratio_) == 1) use_ap = 1;
    else {
      utils::assert(sscanf(name, "pratio@%f", &ratio_) == 1, "BUG");
      use_ap = 0;
    }
  }
  virtual const char *name(void) const { return name_.c_str(); }
};
// \brief ctest
class EvalCTest: public IEvaluator {
 private:
  IEvaluator *base_;
  std::string name_;
 public:
  EvalCTest(IEvaluator *base, const char *name)
      : base_(base), name_(name) {}
  virtual const char *name(void) const {
    return name_.c_str();
  }
  virtual ~EvalCTest(void) { delete base_; }
};
inline IEvaluator* CreateEvaluator(const char *name) {
  if (!strcmp(name, "rmse")) return new EvalRMSE();
  if (!strcmp(name, "error")) return new EvalError();
  if (!strcmp(name, "merror")) return new EvalMatchError();
  if (!strcmp(name, "logloss")) return new EvalLogLoss();
  if (!strcmp(name, "auc")) return new EvalAuc();
  if (!strncmp(name, "ams@", 4)) return new EvalAMS(name);
  if (!strncmp(name, "pre@", 4)) return new EvalPrecision(name);
  if (!strncmp(name, "pratio@", 7)) return new EvalPrecisionRatio(name);
  if (!strncmp(name, "map", 3)) return new EvalMAP(name);
  if (!strncmp(name, "ndcg", 4)) return new EvalNDCG(name);
  if (!strncmp(name, "ct-", 3)) return new EvalCTest(CreateEvaluator(name+3), name);

  utils::error("unknown evaluation metric type: %s", name);
  return NULL;
}

// \brief a set of evaluators
class EvalSet{
 private:
  std::vector<const IEvaluator*> evals_;
 public:
  inline void add_eval(const char *name) {
    for (size_t i = 0; i < evals_.size(); ++i)
      if (!strcmp(name, evals_[i]->name())) return;
    evals_.push_back(CreateEvaluator(name));
  }
  inline size_t size(void) const { return evals_.size(); }
};

}
}
#endif
