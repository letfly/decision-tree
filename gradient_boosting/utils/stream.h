#ifndef UTILS_STREAM_H_
#define UTILS_STREAM_H_
#include <string>

namespace gboost {
namespace utils {
class FileStream{
 private:
  FILE *fp;
 public:
  FileStream(FILE *fp) { this->fp = fp; }
  virtual size_t read(void *ptr, size_t size) { return fread(ptr, size, 1, fp); }
  virtual void write(const void *ptr, size_t size) { fwrite(ptr, size, 1, fp); }
  inline void close(void) { fclose(fp); }
  // \brief binary load a vector 
  // \param out_vec vector to be loaded
  // \return whether load is successfull
  template<typename T>
  inline bool read(std::vector<T> *out_vec) {
    uint64_t sz;
    if (this->read(&sz, sizeof(sz)) == 0) return false;
    out_vec->resize(sz);
    if (sz != 0)
      if (this->read(&(*out_vec)[0], sizeof(T) * sz) == 0) return false;
    return true;
  }
  
  // \brief binary serialize a vector 
  // \param vec vector to be serialized
  template<typename T>
  inline void write(const std::vector<T> &vec) {
    uint64_t sz = static_cast<uint64_t>(vec.size());
    this->write(&sz, sizeof(sz));
    if (sz != 0) this->write(&vec[0], sizeof(T) * sz);
  }
};
}
}
#endif
