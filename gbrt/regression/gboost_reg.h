#include "utils/gboost_stream.h" // IStream, FileStream
#include "regression/gboost_regdata.h" // DMatrix

namespace gboost{
namespace regression{
class RegBoostLearner{
 public:
  inline void SetParam(const char *name, const char *val) {}
  inline void InitModel(void) {}
  inline void LoadModel(utils::IStream &fi) {}
  inline void Predict(std::vector<float> &preds, const DMatrix &data) {}
};
}
}
