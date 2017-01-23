#ifndef LEARNER_EVALUATION_H_
#define LEARNER_EVALUATION_H_
#include <cmath>

namespace gboost {
namespace learner {
// \brief evaluator that evaluates the loss metrics
struct IEvaluator{
  // \return name of metric
  virtual const char *name(void) const = 0;
  // \brief virtual destructor
  virtual ~IEvaluator(void) {}

  // \brief evaluate a specific metric
  // \param preds prediction
  // \param info information, including label etc.
  virtual float eval(const std::vector<float> &preds,
                     const MetaInfo &info) const = 0;
};

template<typename Derived>
struct EvalEWiseBase : public IEvaluator {
  // \brief to be implemented by subclass, 
  //   get evaluation result from one row 
  // \param label label of current instance
  // \param pred prediction value of current instance
  // \param weight weight of current instance
  inline static float eval_row(float label, float pred);
  virtual float eval(const std::vector<float> &preds,
                     const MetaInfo &info) const {
    utils::check(info.labels.size() != 0, "label set cannot be empty");
    utils::check(preds.size() % info.labels.size() == 0,
                 "label and prediction size not match");

    const bst_uint ndata = static_cast<bst_uint>(info.labels.size());

    float sum = 0.0, wsum = 0.0;
    for (bst_uint i = 0; i < ndata; ++i) {
      const float wt = info.get_weight(i);
      sum += Derived::eval_row(info.labels[i], preds[i]) * wt;
      wsum += wt;
    }
    return Derived::get_final(sum, wsum);
  }
  // \brief to be overide by subclas, final trasnformation 
  // \param esum the sum statistics returned by EvalRow
  // \param wsum sum of weight
  inline static float get_final(float esum, float wsum) { return esum / wsum; }
};
// \brief RMSE
struct EvalRMSE : public EvalEWiseBase<EvalRMSE> {
  virtual const char *name(void) const { return "rmse"; }
  inline static float eval_row(float label, float pred) {
    float diff = label - pred;
    return diff * diff;
  }
};
// \brief error
struct EvalError : public EvalEWiseBase<EvalError> {
  virtual const char *name(void) const { return "error"; }
  inline static float eval_row(float label, float pred) {
    // assume label is in [0,1]
    return pred > 0.5f ? 1.0f - label : label;
  }
};
// \brief match error
struct EvalMatchError : public EvalEWiseBase<EvalMatchError> {
  virtual const char *name(void) const { return "merror"; }
  inline static float eval_row(float label, float pred) {
    return static_cast<int>(pred) != static_cast<int>(label);
  }
};
// \brief logloss
struct EvalLogLoss : public EvalEWiseBase<EvalLogLoss> {
  virtual const char *name(void) const { return "logloss"; }
  inline static float eval_row(float y, float py) {
    return - y * log(py) - (1.0f - y) * log(1 - py);
  }
};

inline static bool cmp_first(const std::pair<float, unsigned> &a,
                            const std::pair<float, unsigned> &b) {
  return a.first > b.first;
}
inline static bool cmp_second(const std::pair<float, unsigned> &a,
                             const std::pair<float, unsigned> &b) {
  return a.second > b.second;
}
// \brief Area under curve, for both classification and rank
class EvalAuc : public IEvaluator {
  virtual const char *name(void) const {
    return "auc";
  }
  virtual float eval(const std::vector<float> &preds,
                     const MetaInfo &info) const {
    utils::check(info.labels.size() != 0, "label set cannot be empty");
    utils::check(preds.size() % info.labels.size() == 0,
                 "label size predict size not match");
    std::vector<unsigned> tgptr(2, 0); 
    tgptr[1] = static_cast<unsigned>(info.labels.size());

    const std::vector<unsigned> &gptr = info.group_ptr.size() == 0 ? tgptr : info.group_ptr;
    utils::check(gptr.back() == info.labels.size(),
                 "EvalAuc: group structure must match number of prediction");
    const bst_uint ngroup = static_cast<bst_uint>(gptr.size() - 1);
    // sum statictis
    double sum_auc = 0.0f;
    // each thread takes a local rec
    std::vector< std::pair<float, unsigned> > rec;
    for (bst_uint k = 0; k < ngroup; ++k) {
      rec.clear();
      for (unsigned j = gptr[k]; j < gptr[k + 1]; ++j)
        rec.push_back(std::make_pair(preds[j], j));
      std::sort(rec.begin(), rec.end(), cmp_first);
      // calculate AUC
      double sum_pospair = 0.0;
      double sum_npos = 0.0, sum_nneg = 0.0, buf_pos = 0.0, buf_neg = 0.0;
      for (size_t j = 0; j < rec.size(); ++j) {
        const float wt = info.get_weight(rec[j].second);
        const float ctr = info.labels[rec[j].second];
        // keep bucketing predictions in same bucket
        if (j != 0 && rec[j].first != rec[j - 1].first) {
          sum_pospair += buf_neg * (sum_npos + buf_pos *0.5);
          sum_npos += buf_pos; sum_nneg += buf_neg;
          buf_neg = buf_pos = 0.0f;
        }
        buf_pos += ctr * wt; buf_neg += (1.0f - ctr) * wt;
      }
      sum_pospair += buf_neg * (sum_npos + buf_pos *0.5);
      sum_npos += buf_pos; sum_nneg += buf_neg;
      // check weird conditions
      utils::check(sum_npos > 0.0 && sum_nneg > 0.0,
                   "AUC: the dataset only contains pos or neg samples");
      // this is the AUC
      sum_auc += sum_pospair / (sum_npos*sum_nneg);
    }
    // return average AUC over list
    return static_cast<float>(sum_auc) / ngroup;
  }
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
  virtual float eval(const std::vector<float> &preds,
                     const MetaInfo &info) const {
    const bst_uint ndata = static_cast<bst_uint>(info.labels.size());

    utils::check(info.weights.size() == ndata, "we need weight to evaluate ams");
    std::vector< std::pair<float, unsigned> > rec(ndata);

    for (bst_uint i = 0; i < ndata; ++i)
      rec[i] = std::make_pair(preds[i], i);
    std::sort(rec.begin(), rec.end(), cmp_first);
    unsigned ntop = static_cast<unsigned>(ratio_ * ndata);
    if (ntop == 0) ntop = ndata;
    const double br = 10.0;
    unsigned thresindex = 0;
    double s_tp = 0.0, b_fp = 0.0, tams = 0.0;
    for (unsigned i = 0; i < static_cast<unsigned>(ndata-1) && i < ntop; ++i) {
      const unsigned ridx = rec[i].second;
      const float wt = info.weights[ridx];
      if (info.labels[ridx] > 0.5f) s_tp += wt;
      else b_fp += wt;
      if (rec[i].first != rec[i+1].first) {
        double ams = sqrt(2*((s_tp+b_fp+br) * log(1.0 + s_tp/(b_fp+br)) - s_tp));
        if (tams < ams) {
          thresindex = i;
          tams = ams;
        }
      }
    }
    if (ntop == ndata) {
      printf("\tams-ratio=%g", static_cast<float>(thresindex) / ndata);
      return static_cast<float>(tams);
    } else
      return static_cast<float>(sqrt(2*((s_tp+b_fp+br) * log(1.0 + s_tp/(b_fp+br)) - s_tp)));
  }
};

// \brief Evaluate rank list
class EvalRankList : public IEvaluator {
 private:
  std::string name_;
 protected:
  unsigned topn_;
  bool minus_;
  explicit EvalRankList(const char *name) {
    name_ = name;
    minus_ = false;
    if (sscanf(name, "%*[^@]@%u[-]?", &topn_) != 1) topn_ = UINT_MAX;
    if (name[strlen(name) - 1] == '-') minus_ = true;
  }
  // \return evaluation metric, given the pair_sort record, (pred,label)
  virtual float eval_metric(std::vector< std::pair<float, unsigned> > &pair_sort) const = 0;
 public:
  virtual const char *name(void) const { return name_.c_str(); }
  virtual float eval(const std::vector<float> &preds,
                     const MetaInfo &info) const {
    utils::check(preds.size() == info.labels.size(),
                  "label size predict size not match");
    // quick consistency when group is not available
    std::vector<unsigned> tgptr(2, 0); tgptr[1] = static_cast<unsigned>(preds.size());
    const std::vector<unsigned> &gptr = info.group_ptr.size() == 0 ? tgptr : info.group_ptr;
    utils::assert(gptr.size() != 0, "must specify group when constructing rank file");
    utils::assert(gptr.back() == preds.size(),
                   "EvalRanklist: group structure must match number of prediction");
    const bst_uint ngroup = static_cast<bst_uint>(gptr.size() - 1);
    // sum statistics
    double sum_metric = 0.0f;
    // each thread takes a local rec
    std::vector< std::pair<float, unsigned> > rec;
    for (bst_uint k = 0; k < ngroup; ++k) {
      rec.clear();
      for (unsigned j = gptr[k]; j < gptr[k + 1]; ++j)
        rec.push_back(std::make_pair(preds[j], static_cast<int>(info.labels[j])));
      sum_metric += this->eval_metric(rec);
    }
    return static_cast<float>(sum_metric) / ngroup;
  }
};
// \brief Precison at N, for both classification and rank
class EvalPrecision : public EvalRankList{
 public:
  explicit EvalPrecision(const char *name) : EvalRankList(name) {}
  virtual float eval_metric(std::vector< std::pair<float, unsigned> > &rec) const {
    // calculate Preicsion
    std::sort(rec.begin(), rec.end(), cmp_first);
    unsigned nhit = 0;
    for (size_t j = 0; j < rec.size() && j < this->topn_; ++j)
      nhit += (rec[j].second != 0);
    return static_cast<float>(nhit) / topn_;
  }
};
// \brief Precison at N, for both classification and rank
class EvalMAP : public EvalRankList {
 public:
  explicit EvalMAP(const char *name) : EvalRankList(name) {}
  virtual float eval_metric(std::vector< std::pair<float, unsigned> > &rec) const {
    std::sort(rec.begin(), rec.end(), cmp_first);
    unsigned nhits = 0;
    double sumap = 0.0;
    for (size_t i = 0; i < rec.size(); ++i) {
      if (rec[i].second != 0) {
        nhits += 1;
        if (i < this->topn_) sumap += static_cast<float>(nhits) / (i+1);
      }
    }
    if (nhits != 0) {
      sumap /= nhits;
      return static_cast<float>(sumap);
    } else {
      if (minus_) return 0.0f;
      else return 1.0f;
    }
  }
};
// \brief NDCG
class EvalNDCG : public EvalRankList{
 public:
  explicit EvalNDCG(const char *name) : EvalRankList(name) {}
  inline float calc_DCG(const std::vector< std::pair<float, unsigned> > &rec) const {
    double sumdcg = 0.0;
    for (size_t i = 0; i < rec.size() && i < this->topn_; ++i) {
      const unsigned rel = rec[i].second;
      if (rel != 0) { 
        sumdcg += ((1 << rel) - 1) / log(i + 2.0);
      }
    }
    return static_cast<float>(sumdcg);
  }
  virtual float eval_metric(std::vector< std::pair<float, unsigned> > &rec) const {
    std::stable_sort(rec.begin(), rec.end(), cmp_first);
    float dcg = this->calc_DCG(rec);
    std::stable_sort(rec.begin(), rec.end(), cmp_second);
    float idcg = this->calc_DCG(rec);
    if (idcg == 0.0f) {
      if (minus_) return 0.0f;
      else return 1.0f;
    }
    return dcg/idcg;
  }
};

// \brief precision with cut off at top percentile
class EvalPrecisionRatio : public IEvaluator{
 private:
  int use_ap;
  float ratio_;
  std::string name_;

  inline double calc_pre_ratio(const std::vector< std::pair<float, unsigned> >& rec, const MetaInfo &info) const {
    size_t cutoff = static_cast<size_t>(ratio_ * rec.size());
    double wt_hit = 0.0, wsum = 0.0, wt_sum = 0.0;
    for (size_t j = 0; j < cutoff; ++j) {
      const float wt = info.get_weight(j);
      wt_hit += info.labels[rec[j].second] * wt;
      wt_sum += wt;
      wsum += wt_hit / wt_sum;
    }
    if (use_ap != 0) return wsum / cutoff;
    else return wt_hit / wt_sum;
  }
 public:
  explicit EvalPrecisionRatio(const char *name) : name_(name) {
    if (sscanf(name, "apratio@%f", &ratio_) == 1) use_ap = 1;
    else {
      utils::assert(sscanf(name, "pratio@%f", &ratio_) == 1, "BUG");
      use_ap = 0;
    }
  }
  virtual const char *name(void) const { return name_.c_str(); }
  virtual float eval(const std::vector<float> &preds,
                     const MetaInfo &info) const {
    utils::check(info.labels.size() != 0, "label set cannot be empty");    
    utils::assert(preds.size() % info.labels.size() == 0,
                  "label size predict size not match");
    std::vector< std::pair<float, unsigned> > rec;
    for (size_t j = 0; j < info.labels.size(); ++j)
      rec.push_back(std::make_pair(preds[j], static_cast<unsigned>(j)));
    std::sort(rec.begin(), rec.end(), cmp_first);
    double pratio = calc_pre_ratio(rec, info);
    return static_cast<float>(pratio);
  }
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
  virtual float eval(const std::vector<float> &preds,
                     const MetaInfo &info) const {
    utils::check(preds.size() % info.labels.size() == 0,
                 "label and prediction size not match");
    size_t ngroup = preds.size() / info.labels.size() - 1;
    const unsigned ndata = static_cast<unsigned>(info.labels.size());
    utils::check(ngroup > 1, "pred size does not meet requirement");
    utils::check(ndata == info.info.fold_index.size(), "need fold index");
    double wsum = 0.0;
    for (size_t k = 0; k < ngroup; ++k) {
      std::vector<float> tpred;
      MetaInfo tinfo;
      for (unsigned i = 0; i < ndata; ++i) {
        if (info.info.fold_index[i] == k) {
          tpred.push_back(preds[i + (k + 1) * ndata]);
          tinfo.labels.push_back(info.labels[i]);
          tinfo.weights.push_back(info.get_weight(i));
        }        
      }
      wsum += base_->eval(tpred, tinfo);
    }
    return static_cast<float>(wsum / ngroup);
  }
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
  inline size_t size(void) const {
    return evals_.size();
  }

  inline std::string eval(const char *evname,
                          const std::vector<float> &preds,
                          const MetaInfo &info) const {
    std::string result = "";
    for (size_t i = 0; i < evals_.size(); ++i) {
      float res = evals_[i]->eval(preds, info);
      char tmp[1024];
      utils::sprintf(tmp, sizeof(tmp), "\t%s-%s:%f", evname, evals_[i]->name(), res);
      result += tmp;
    }
    return result;
  }
};

}
}
#endif
