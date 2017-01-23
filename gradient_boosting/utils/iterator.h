#ifndef UTILS_ITERATOR_H_
#define UTILS_ITERATOR_H_
namespace gboost {
namespace utils {
// \brief iterator interface
// \tparam DType data type
template<typename DType>
class IIterator {
 public:
  // \brief move to next item
  virtual bool next(void) = 0;
  // \brief get current data
  virtual const DType &value(void) const = 0;
  // \brief set before first of the item 
  virtual void before_first(void) = 0;
};
}
}
#endif
