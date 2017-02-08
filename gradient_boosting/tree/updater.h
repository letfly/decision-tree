#ifndef TREE_UPDATER_H_
#define TREE_UPDATER_H_
#include "tree/model.h" // RegTree

namespace gboost {
namespace tree {
// \brief interface of tree update module, that performs update of a tree
class IUpdater {
 public:
  // \brief set parameters from outside
  // \param name name of the parameter
  // \param val  value of the parameter
  virtual void set_param(const char *name, const char *val) = 0;
  // \brief peform update to the tree models
  // \param gpair the gradient pair statistics of the data
  // \param p_fmat feature matrix that provide access to features
  // \param info extra side information that may be need, such as root index
  // \param trees pointer to the trese to be updated, upater will change the content of the tree
  //   note: all the trees in the vector are updated, with the same statistics, 
  //         but maybe different random seeds, usually one tree is passed in at a time, 
  //         there can be multiple trees when we train random forest style model
  virtual void update(const std::vector<bst_gpair> &gpair,
                      IFMatrix *p_fmat,
                      const BoosterInfo &info,
                      const std::vector<RegTree*> &trees) = 0;
};

// \brief pruner that prunes a tree after growing finishs
class TreePruner: public IUpdater {
 private:
  int silent;
  // \brief do prunning of a tree
  inline void do_prune(RegTree &tree) {
    int npruned = 0;
    // initialize auxiliary statistics
    for (int nid = 0; nid < tree.param.num_nodes; ++nid) {
      tree.stat(nid).leaf_child_cnt = 0;
    }
    for (int nid = 0; nid < tree.param.num_nodes; ++nid) {
      if (tree[nid].is_leaf()) {
        npruned = this->try_prune_leaf(tree, nid, tree.get_depth(nid), npruned);
      }
    }
    if (silent == 0) {
      printf("tree prunning end, %d roots, %d extra nodes, %d pruned nodes ,max_depth=%d\n",
                    tree.param.num_roots, tree.num_extra_nodes(), npruned, tree.max_depth());
    }
  }
  // try to prune off current leaf
  inline int try_prune_leaf(RegTree &tree, int nid, int depth, int npruned) {
    if (tree[nid].is_root()) return npruned;
    int pid = tree[nid].parent();
    RTreeNodeStat &s = tree.stat(pid);
    ++s.leaf_child_cnt;
    if (s.leaf_child_cnt >= 2) {
      // need to be pruned
      tree.change_to_leaf(pid, 0.3f * s.base_weight);
      // tail recursion
      return this->try_prune_leaf(tree, pid, depth - 1, npruned+2);
    } else {
      return npruned;
    }
  }
 public:
  virtual ~TreePruner(void) {}
  // set training parameter
  virtual void set_param(const char *name, const char *val) {
    if (!strcmp(name, "silent")) silent = atoi(val);
  }
  // update the tree, do pruning
  virtual void update(const std::vector<bst_gpair> &gpair,
                      IFMatrix *p_fmat,
                      const BoosterInfo &info,
                      const std::vector<RegTree*> &trees) {
    // rescale learning rate according to size of trees
    for (size_t i = 0; i < trees.size(); ++i) {
      this->do_prune(*trees[i]);
    }
  }
};

// \brief pruner that prunes a tree after growing finishs
template<typename TStats>
class ColMaker: public IUpdater {
 public:
  // set training parameter
  virtual void set_param(const char *name, const char *val) {}
  virtual void update(const std::vector<bst_gpair> &gpair,
                      IFMatrix *p_fmat,
                      const BoosterInfo &info,
                      const std::vector<RegTree*> &trees) {}
};
// \brief core statistics used for tree construction
struct GradStats {
};

// \brief create a updater based on name 
// \param name name of updater
// \return return the updater instance
IUpdater* CreateUpdater(const char *name) {
  if (!strcmp(name, "prune")) return new TreePruner();
  if (!strcmp(name, "grow_colmaker")) return new ColMaker<GradStats>();
  //utils::error("unknown updater:%s", name);
  return NULL;
}

}
}
#endif
