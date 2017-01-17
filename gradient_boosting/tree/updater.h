#ifndef TREE_UPDATER_H_
#define TREE_UPDATER_H_
namespace gboost {
namespace tree {
// \brief interface of tree update module, that performs update of a tree
class IUpdater {
 public:
  // \brief set parameters from outside
  // \param name name of the parameter
  // \param val  value of the parameter
  virtual void set_param(const char *name, const char *val) = 0;
};

}
}
#endif
