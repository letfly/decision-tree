#ifndef GBOOST_REGDATA_H_
#define GBOOST_REGDATA_H_
#include <fstream> // ifstream
#include "booster/gboost_data.h" // FMatrixS, Clear, AddRow, NumRow, NumEntry
#include "utils/gboost_utils.h" // FopenCheck

namespace gboost {
namespace regression {
struct DMatrix {
 private:
  std::vector<float> labels;
  inline void UpdateInfo(void) {
    this->num_feature = 0;
    for (size_t i = 0; i < data.NumRow(); ++i) {
      booster::FMatrixS::Line sp = data[i];
      for (unsigned j = 0; j < sp.len; ++j)
        if (num_feature <= sp.findex[j])
          num_feature = sp.findex[j]+1;
    }
  }
 public:
  // Maximum feature dimension
  unsigned num_feature;
  // Feature data
  booster::FMatrixS data;
  inline void LoadText(const char *fname, bool silent = false) {
    data.Clear();
    //FILE *file = utils::FopenCheck(fname, "r");
    std::ifstream file(fname);
    float label; bool init = true;
    char tmp[1024];
    std::string line;
    std::vector<booster::bst_uint> findex;
    std::vector<booster::bst_float> fvalue;

    while (getline(file, line)) {
      unsigned index; float value;
      printf("tmp=%s %d %d %f", tmp, sscanf(tmp, "%u:%f", &index, &value), index, value);
      if (sscanf(tmp, "%u:%f", &index, &value) == 2) {
        findex.push_back(index); fvalue.push_back(value);
      } else {
        if (!init) {
          labels.push_back(label);
          data.AddRow(findex, fvalue);
        }
        findex.clear(); fvalue.clear();
        utils::Assert(sscanf(tmp, "%f", &label) == 1, "invalid format");
        init = false;
      }
    }
    labels.push_back(label);
    data.AddRow(findex, fvalue);

    this->UpdateInfo();
    if (!silent)
      printf("%ux%u matrix with %lu entries is loaded from %s\n",
        (unsigned)labels.size(), num_feature, (unsigned long)data.NumEntry(), fname);
    file.close();
  }
};
}
}
#endif
