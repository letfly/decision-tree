namespace gboost {
namespace utils {
class IStream {
 public:
  virtual void write(const void *ptr, size_t size) = 0;
};
class FileStream: public IStream {
 private:
  FILE *fp;
 public:
  FileStream(FILE *fp) { this->fp = fp; }
  virtual void write(const void *ptr, size_t size) { fwrite(ptr, size, 1, fp); }
  inline void close(void) { fclose(fp); }
};
}
}
