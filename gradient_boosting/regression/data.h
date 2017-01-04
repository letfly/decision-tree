#ifndef REG_DATA_H_
#define REG_DATA_H_
#include <fstream> // ifstream
#include "booster/data.h" // FMatrixS, clear, add_row, num_row, operator, num_entry
#include "utils/utils.h" // fopen_check

namespace gboost {
namespace regression {
struct DMatrix {
 private:
  std::vector<float> labels;
  inline void update_info(void) {
    this->num_feature = 0;
    for (size_t i = 0; i < data.num_row(); ++i) {
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
  inline void load_text(const char *fname, bool silent = false) {
    data.clear();
    FILE *file = utils::fopen_check(fname, "r");
    //std::ifstream file(fname);
    float label; bool init = true;
    char tmp[1024];
    std::string line;
    std::vector<booster::bst_uint> findex;
    std::vector<booster::bst_float> fvalue;

    while (fscanf(file, "%s", tmp) == 1) {
      unsigned index; float value;
      //printf("tmp=%s %d %d %f", tmp, sscanf(tmp, "%u:%f", &index, &value), index, value);
      if (sscanf(tmp, "%u:%f", &index, &value) == 2) {
        findex.push_back(index); fvalue.push_back(value);
      } else {
        if (!init) {
          labels.push_back(label);
          data.add_row(findex, fvalue);
        }
        findex.clear(); fvalue.clear();
        utils::assert(sscanf(tmp, "%f", &label) == 1, "invalid format");
        init = false;
      }
    }
    labels.push_back(label);
    data.add_row(findex, fvalue);

    this->update_info();
    if (!silent)
      printf("%ux%u matrix with %lu entries is loaded from %s\n",
        (unsigned)labels.size(), num_feature, (unsigned long)data.num_entry(), fname);
    fclose(file);
  }
  inline size_t size() const { return labels.size(); }
};
}
}
#endif
