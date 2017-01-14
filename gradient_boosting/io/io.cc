#include "io/simple_dmatrix.h" // DMatrixSimple
#include "learner/dmatrix.h"
#include "utils/stream.h"
#include "utils/utils.h"

namespace gboost {
namespace io {
learner::DMatrix* load_data_matrix(const char *fname, bool silent, bool savebuffer) {
  printf("%s", fname);
  int magic;
  utils::FileStream fs(utils::fopen_check(fname, "rb"));
  utils::check(fs.read(&magic, sizeof(magic))!=0, "invalid input file format");

  if (magic == DMatrixSimple::kMagic) {}
  fs.close();
  DMatrixSimple *dmat = new DMatrixSimple();
  dmat->cache_load(fname, silent, savebuffer);
  return dmat;
}
}
}
