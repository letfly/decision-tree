#ifndef IO_IO_H_
#define IO_IO_H_
#include "learner/dmatrix.h"

namespace gboost {
namespace io {
learner::DMatrix *load_data_matrix(const char *fname, bool silent=false, bool savebuffer=true);
}
}
#endif
