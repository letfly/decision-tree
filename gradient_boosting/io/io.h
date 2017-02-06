#ifndef IO_IO_H_
#define IO_IO_H_
#include "learner/dmatrix.h"

namespace gboost {
namespace io {
// \brief load DataMatrix from stream
// \param fname file name to be loaded
// \param silent whether print message during loading
// \param savebuffer whether temporal buffer the file if the file is in text format
// \return a loaded DMatrix
learner::DMatrix *load_data_matrix(const char *fname, bool silent=false, bool savebuffer=true);
}
}
#endif
