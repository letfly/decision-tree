#ifndef TREE_UPDATER_H_
#define TREE_UPDATER_H_
#include "tree/model.h" // RegTree
#include "utils/random.h" // sample_binary, shuffle

namespace gboost {
namespace tree {
// \brief training parameters for regression tree
class TrainParam {
 private:
  // functions for L1 cost
  inline static double threshold_L1(double w, double lambda) {
    if (w > +lambda) return w - lambda;
    if (w < -lambda) return w + lambda;
    return 0.0;
  }
 public:
  // learning step size for a time
  float learning_rate;
  // whether we want to do subsample
  float subsample;
  // whether to subsample columns during tree construction
  float colsample_bytree;
  // whether to subsample columns each split, in each level
  float colsample_bylevel;
  // default direction choice
  int default_direction;
  // speed optimization for dense column
  float opt_dense_col;
  // maximum depth of a tree
  int max_depth;

  // minimum amount of hessian(weight) allowed in a child
  float min_child_weight;
  // L1 regularization factor
  float reg_alpha;
  // L2 regularization factor
  float reg_lambda;
  // \brief constructor
  TrainParam(void) {
    learning_rate = 0.3f;
    subsample = 1.0f;
    colsample_bytree = 1.0f;
    colsample_bylevel = 1.0f;
    default_direction = 0;
    opt_dense_col = 1.0f;
    max_depth = 6;
    min_child_weight = 1.0f;
    reg_alpha = 0.0f;
    reg_lambda = 1.0f;
  }
  // \brief whether need forward small to big search: default right
  inline bool need_forward_search(float col_density = 0.0f) const {
    return this->default_direction == 2 ||
        (this->default_direction == 0 && (col_density < this->opt_dense_col));
  }
  // \brief whether need backward big to small search: default left
  inline bool need_backward_search(float col_density = 0.0f) const {
    return this->default_direction != 2;
  }
  // calculate the cost of loss function
  inline double calc_gain(double sum_grad, double sum_hess) const {
    if (sum_hess < min_child_weight) return 0.0;
    if (reg_alpha == 0.0f) {
      return sqrt(sum_grad) / (sum_hess + reg_lambda);
    } else {
      return sqrt(threshold_L1(sum_grad, reg_alpha)) / (sum_hess + reg_lambda); 
    }
  }
  // calculate weight given the statistics
  inline double calc_weight(double sum_grad, double sum_hess) const {
    if (sum_hess < min_child_weight) return 0.0;
    if (reg_alpha == 0.0f) {
      return -sum_grad / (sum_hess + reg_lambda);
    } else {
      return -threshold_L1(sum_grad, reg_alpha) / (sum_hess + reg_lambda);
    }
  }
  inline void set_param(const char *name, const char *val) {
    if (!strcmp(name, "max_depth")) max_depth = atoi(val);
  }
};

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

// \brief statistics that is helpful to store 
//   and represent a split solution for the tree
struct SplitEntry{
  // \brief loss change after split this node
  bst_float loss_chg;
  // \brief split index
  unsigned sindex;
  // \return feature index to split on
  inline unsigned split_index(void) const {
    return sindex & ((1U << 31) - 1U);
  }
  // \brief split value
  float split_value;
  // \brief update the split entry, replace it if e is better
  // \param loss_chg loss reduction of new candidate
  // \param split_index feature index to split on
  // \param split_value the split point
  // \param default_left whether the missing value goes to left
  // \return whether the proposed split is better and can replace current split
  inline bool update(bst_float new_loss_chg, unsigned split_index,
                     float new_split_value, bool default_left) {
    if (this->need_replace(new_loss_chg, split_index)) {
      this->loss_chg = new_loss_chg;
      if (default_left) split_index |= (1U << 31);
      this->sindex = split_index;
      this->split_value = new_split_value;
      return true;
    } else {
      return false;
    }
  }
  // \brief decides whether a we can replace current entry with the statistics given 
  //   This function gives better priority to lower index when loss_chg equals
  //    not the best way, but helps to give consistent result during multi-thread execution
  // \param loss_chg the loss reduction get through the split
  // \param split_index the feature index where the split is on 
  inline bool need_replace(bst_float new_loss_chg, unsigned split_index) const {
    if (this->split_index() <= split_index) {
      return new_loss_chg > this->loss_chg;
    } else {
      return !(this->loss_chg > new_loss_chg);
    }
  }
  // \brief update the split entry, replace it if e is better
  // \param e candidate split solution
  // \return whether the proposed split is better and can replace current split
  inline bool update(const SplitEntry &e) {
    if (this->need_replace(e.loss_chg, e.split_index())) {
      this->loss_chg = e.loss_chg;
      this->sindex = e.sindex;
      this->split_value = e.split_value;
      return true;
    } else {
      return false;
    }
  }
  // \return whether missing value goes to left branch
  inline bool default_left(void) const {
    return (sindex >> 31) != 0;
  }
};

// \brief pruner that prunes a tree after growing finishs
class TreePruner: public IUpdater {
 private:
  // training parameter
  TrainParam param;
  int silent;
  // \brief do prunning of a tree
  inline void do_prune(RegTree &tree) {
    int npruned = 0;
    // initialize auxiliary statistics
    for (int nid = 0; nid < tree.param.num_nodes; ++nid)
      tree.stat(nid).leaf_child_cnt = 0;
    for (int nid = 0; nid < tree.param.num_nodes; ++nid)
      if (tree[nid].is_leaf())
        npruned = this->try_prune_leaf(tree, nid, tree.get_depth(nid), npruned);
    if (silent == 0)
      printf("tree prunning end, %d roots, %d extra nodes, %d pruned nodes ,max_depth=%d\n",
                    tree.param.num_roots, tree.num_extra_nodes(), npruned, tree.max_depth());
  }
  // try to prune off current leaf
  inline int try_prune_leaf(RegTree &tree, int nid, int depth, int npruned) {
    if (tree[nid].is_root()) return npruned;
    int pid = tree[nid].parent();
    RTreeNodeStat &s = tree.stat(pid);
    ++s.leaf_child_cnt;
    if (s.leaf_child_cnt >= 2) {
      // need to be pruned
      tree.change_to_leaf(pid, param.learning_rate * s.base_weight);
      // tail recursion
      return this->try_prune_leaf(tree, pid, depth - 1, npruned+2);
    } else return npruned;
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
    for (size_t i = 0; i < trees.size(); ++i) { // error trees.size()
      this->do_prune(*trees[i]);
    }
  }
};

// \brief pruner that prunes a tree after growing finishs
template<typename TStats>
class ColMaker: public IUpdater {
 private:
  // training parameter
  TrainParam param;
  // \brief per thread x per node entry to store tmp data
  struct ThreadEntry {
    // constructor
    explicit ThreadEntry(const TrainParam &param) 
        : stats(param) {
    }
    // \brief statistics of data
    TStats stats;
    // \brief current best solution
    SplitEntry best;
    // \brief last feature value scanned
    float last_fvalue;
  };
  struct NodeEntry {
    // \brief weight calculated related to current data
    float weight;
    // \brief current best solution
    SplitEntry best;
    // constructor
    explicit NodeEntry(const TrainParam &param)
        : stats(param), root_gain(0.0f), weight(0.0f){
    }
    // \brief statics for node entry
    TStats stats;
    // \brief loss of this node, without split
    bst_float root_gain;
  };
  // actual builder that runs the algorithm
  class Builder{
   private:
    const TrainParam &param;
    // Instance Data: current node position in the tree of each instance
    std::vector<int> position;
    // PerThread x PerTreeNode: statistics for per thread construction
    std::vector< std::vector<ThreadEntry> > stemp;
    // initialize temp data structure
    inline void init_data(const std::vector<bst_gpair> &gpair,
                         const IFMatrix &fmat,
                         const std::vector<unsigned> &root_index, const RegTree &tree) {
      utils::assert(tree.param.num_nodes == tree.param.num_roots, "ColMaker: can only grow new tree");
      const std::vector<bst_uint> &rowset = fmat.buffered_rowset();
      {// setup position
        position.resize(gpair.size());
        if (root_index.size() == 0) {
          for (size_t i = 0; i < rowset.size(); ++i) {
            position[rowset[i]] = 0;
          }
        } else {
          for (size_t i = 0; i < rowset.size(); ++i) {
            const bst_uint ridx = rowset[i];
            position[ridx] = root_index[ridx];
            utils::assert(root_index[ridx] < (unsigned)tree.param.num_roots, "root index exceed setting");
          }
        }
        // mark delete for the deleted datas
        for (size_t i = 0; i < rowset.size(); ++i) {
          const bst_uint ridx = rowset[i];
          if (gpair[ridx].hess < 0.0f) position[ridx] = -1;
        }
        // mark subsample
        if (param.subsample < 1.0f) {
          for (size_t i = 0; i < rowset.size(); ++i) {
            const bst_uint ridx = rowset[i];
            if (gpair[ridx].hess < 0.0f) continue;
            if (random::sample_binary(param.subsample) == 0) position[ridx] = -1;
          }
        }
      }    
      {
        // initialize feature index
        unsigned ncol = static_cast<unsigned>(fmat.num_col());
        for (unsigned i = 0; i < ncol; ++i) {
          if (fmat.get_col_size(i) != 0) {
            feat_index.push_back(i);
          }
        }
        unsigned n = static_cast<unsigned>(param.colsample_bytree * feat_index.size());
        random::shuffle(feat_index);
        utils::check(n > 0, "colsample_bytree is too small that no feature can be included");
        feat_index.resize(n);
      }
      {// setup temp space for each thread
        this->nthread = 1;
        // reserve a small space
        stemp.clear();
        stemp.resize(this->nthread, std::vector<ThreadEntry>());
        for (size_t i = 0; i < stemp.size(); ++i) {
          stemp[i].clear(); stemp[i].reserve(256);
        }
        snode.reserve(256);
      }
      {// expand query
        qexpand_.reserve(256); qexpand_.clear();
        for (int i = 0; i < tree.param.num_roots; ++i) {
          qexpand_.push_back(i);
        }
      }
    }
    // Per feature: shuffle index of each feature index
    std::vector<bst_uint> feat_index;
    // \brief TreeNode Data: statistics for each constructed node
    std::vector<NodeEntry> snode;
    // \brief queue of nodes to be expanded
    std::vector<int> qexpand_;
    // number of omp thread used during training
    int nthread;
    // \brief initialize the base_weight, root_gain, and NodeEntry for all the new nodes in qexpand
    inline void init_new_node(const std::vector<int> &qexpand,
                              const std::vector<bst_gpair> &gpair,
                              const IFMatrix &fmat,
                              const BoosterInfo &info,
                              const RegTree &tree) {
      {// setup statistics space for each tree node
        for (size_t i = 0; i < stemp.size(); ++i) {
          stemp[i].resize(tree.param.num_nodes, ThreadEntry(param));
        }
        snode.resize(tree.param.num_nodes, NodeEntry(param));
      }
      const std::vector<bst_uint> &rowset = fmat.buffered_rowset();
      // setup position
      const bst_uint ndata = static_cast<bst_uint>(rowset.size());
      for (bst_uint i = 0; i < ndata; ++i) {
        const bst_uint ridx = rowset[i];
        const int tid = 0;
        if (position[ridx] < 0) continue;
        stemp[tid][position[ridx]].stats.add(gpair, info, ridx);
      }
      // sum the per thread statistics together
      for (size_t j = 0; j < qexpand.size(); ++j) {
        const int nid = qexpand[j];
        TStats stats(param);
        for (size_t tid = 0; tid < stemp.size(); ++tid) {
          stats.add(stemp[tid][nid].stats);
        }
        // update node statistics
        snode[nid].stats = stats;
        snode[nid].root_gain = static_cast<float>(stats.calc_gain(param));
        snode[nid].weight = static_cast<float>(stats.calc_weight(param));
      }
    }
    // find splits at current level, do split per level
    inline void find_split(int depth,
                          const std::vector<int> &qexpand,
                          const std::vector<bst_gpair> &gpair,
                          IFMatrix *p_fmat,
                          const BoosterInfo &info,
                          RegTree *p_tree) {
      std::vector<bst_uint> feat_set = feat_index;
      if (param.colsample_bylevel != 1.0f) {
        random::shuffle(feat_set);
        unsigned n = static_cast<unsigned>(param.colsample_bylevel * feat_index.size());
        utils::check(n > 0, "colsample_bylevel is too small that no feature can be included");
        feat_set.resize(n);
      }
      utils::IIterator<ColBatch> *iter = p_fmat->col_iterator(feat_set);
      while (iter->next()) {
        const ColBatch &batch = iter->value();
        // start enumeration
        const bst_uint nsize = static_cast<bst_uint>(batch.size);
        const int batch_size = std::max(static_cast<int>(nsize / this->nthread / 32), 1);
        for (bst_uint i = 0; i < nsize; ++i) {
          const bst_uint fid = batch.col_index[i];
          const int tid = 0;
          const ColBatch::Inst c = batch[i];
          if (param.need_forward_search(p_fmat->get_col_density(fid))) {
            this->enumerate_split(c.data, c.data + c.length, +1,
                                 fid, gpair, info, stemp[tid]);
          }
          if (param.need_backward_search(p_fmat->get_col_density(fid))) {
            this->enumerate_split(c.data + c.length - 1, c.data - 1, -1,
                                 fid, gpair, info, stemp[tid]);
          }
        }
      }
      // after this each thread's stemp will get the best candidates, aggregate results
      for (size_t i = 0; i < qexpand.size(); ++i) {
        const int nid = qexpand[i];
        NodeEntry &e = snode[nid];
        for (int tid = 0; tid < this->nthread; ++tid) {
          e.best.update(stemp[tid][nid].best);
        }
        // now we know the solution in snode[nid], set split
        if (e.best.loss_chg > rt_eps) {
          p_tree->add_childs(nid);
          (*p_tree)[nid].set_split(e.best.split_index(), e.best.split_value, e.best.default_left());
        } else {
          (*p_tree)[nid].set_leaf(e.weight * param.learning_rate);
        }
      }
    }
   public:
    // constructor
    explicit Builder(const TrainParam &param) : param(param) {}
    // reset position of each data points after split is created in the tree
    inline void reset_position(const std::vector<int> &qexpand, IFMatrix *p_fmat, const RegTree &tree) {
      const std::vector<bst_uint> &rowset = p_fmat->buffered_rowset();
      // step 1, set default direct nodes to default, and leaf nodes to -1
      const bst_uint ndata = static_cast<bst_uint>(rowset.size());
      for (bst_uint i = 0; i < ndata; ++i) {
        const bst_uint ridx = rowset[i];
        const int nid = position[ridx];
        if (nid >= 0) {
          if (tree[nid].is_leaf()) {
            position[ridx] = -1;
          } else {
            // push to default branch, correct latter
            position[ridx] = tree[nid].default_left() ? tree[nid].cleft(): tree[nid].cright();
          }
        }
      }
      // step 2, classify the non-default data into right places
      std::vector<unsigned> fsplits;
      for (size_t i = 0; i < qexpand.size(); ++i) {
        const int nid = qexpand[i];
        if (!tree[nid].is_leaf()) fsplits.push_back(tree[nid].split_index());
      }
      std::sort(fsplits.begin(), fsplits.end());
      fsplits.resize(std::unique(fsplits.begin(), fsplits.end()) - fsplits.begin());

      utils::IIterator<ColBatch> *iter = p_fmat->col_iterator(fsplits);
      while (iter->next()) {
        const ColBatch &batch = iter->value();
        for (size_t i = 0; i < batch.size; ++i) {
          ColBatch::Inst col = batch[i];
          const bst_uint fid = batch.col_index[i];
          const bst_uint ndata = static_cast<bst_uint>(col.length);
          for (bst_uint j = 0; j < ndata; ++j) {
            const bst_uint ridx = col[j].index;
            const float fvalue = col[j].fvalue;
            int nid = position[ridx];
            if (nid == -1) continue;
            // go back to parent, correct those who are not default
            nid = tree[nid].parent();
            if (tree[nid].split_index() == fid) {
              if (fvalue < tree[nid].split_cond()) {
                position[ridx] = tree[nid].cleft();
              } else {
                position[ridx] = tree[nid].cright();
              }
            }
          }
        }
      }
    }
    // \brief update queue expand add in new leaves
    inline void update_queue_expand(const RegTree &tree, std::vector<int> *p_qexpand) {
      std::vector<int> &qexpand = *p_qexpand;
      std::vector<int> newnodes;
      for (size_t i = 0; i < qexpand.size(); ++i) {
        const int nid = qexpand[i];
        if (!tree[ nid ].is_leaf()) {
          newnodes.push_back(tree[nid].cleft());
          newnodes.push_back(tree[nid].cright());
        }
      }
      // use new nodes for qexpand
      qexpand = newnodes;
    }
    // update one tree, growing
    virtual void update(const std::vector<bst_gpair> &gpair,
                        IFMatrix *p_fmat,
                        const BoosterInfo &info,
                        RegTree *p_tree) {
      this->init_data(gpair, *p_fmat, info.root_index, *p_tree);
      this->init_new_node(qexpand_, gpair, *p_fmat, info, *p_tree);
      for (int depth = 0; depth < param.max_depth; ++depth) {
        this->find_split(depth, qexpand_, gpair, p_fmat, info, p_tree);
        this->reset_position(qexpand_, p_fmat, *p_tree);
        this->update_queue_expand(*p_tree, &qexpand_);
        this->init_new_node(qexpand_, gpair, *p_fmat, info, *p_tree);
        // if nothing left to be expand, break
        if (qexpand_.size() == 0) break;
      }
      // set all the rest expanding nodes to leaf
      for (size_t i = 0; i < qexpand_.size(); ++i) {
        const int nid = qexpand_[i];
        (*p_tree)[nid].set_leaf(snode[nid].weight * param.learning_rate);
      }
      // remember auxiliary statistics in the tree node
      for (int nid = 0; nid < p_tree->param.num_nodes; ++nid) {
        p_tree->stat(nid).loss_chg = snode[nid].best.loss_chg;
        p_tree->stat(nid).base_weight = snode[nid].weight;
        p_tree->stat(nid).sum_hess = static_cast<float>(snode[nid].stats.sum_hess);
        snode[nid].stats.SetLeafVec(param, p_tree->leafvec(nid));
      }
    }
    // enumerate the split values of specific feature
    inline void enumerate_split(const ColBatch::Entry *begin,
                               const ColBatch::Entry *end,
                               int d_step,
                               bst_uint fid,
                               const std::vector<bst_gpair> &gpair,
                               const BoosterInfo &info,
                               std::vector<ThreadEntry> &temp) {
      const std::vector<int> &qexpand = qexpand_;
      // clear all the temp statistics
      for (size_t j = 0; j < qexpand.size(); ++j) {
        temp[qexpand[j]].stats.clear();
      }
      // left statistics
      TStats c(param);
      for(const ColBatch::Entry *it = begin; it != end; it += d_step) {
        const bst_uint ridx = it->index;
        const int nid = position[ridx];
        if (nid < 0) continue;
        // start working
        const float fvalue = it->fvalue;
        // get the statistics of nid
        ThreadEntry &e = temp[nid];
        // test if first hit, this is fine, because we set 0 during init
        if (e.stats.empty()) {
          e.stats.add(gpair, info, ridx);
          e.last_fvalue = fvalue;
        } else {
          // try to find a split
          if (std::abs(fvalue - e.last_fvalue) > rt_2eps && e.stats.sum_hess >= param.min_child_weight) {
            c.set_substract(snode[nid].stats, e.stats);
            if (c.sum_hess >= param.min_child_weight) {
              bst_float loss_chg = static_cast<bst_float>(e.stats.calc_gain(param) + c.calc_gain(param) - snode[nid].root_gain);
              e.best.update(loss_chg, fid, (fvalue + e.last_fvalue) * 0.5f, d_step == -1);
            }
          }
          // update the statistics
          e.stats.add(gpair, info, ridx);
          e.last_fvalue = fvalue;
        }
      }
      // finish updating all statistics, check if it is possible to include all sum statistics
      for (size_t i = 0; i < qexpand.size(); ++i) {
        const int nid = qexpand[i];
        ThreadEntry &e = temp[nid];
        c.set_substract(snode[nid].stats, e.stats);
        if (e.stats.sum_hess >= param.min_child_weight && c.sum_hess >= param.min_child_weight) {
          bst_float loss_chg = static_cast<bst_float>(e.stats.calc_gain(param) + c.calc_gain(param) - snode[nid].root_gain);
          const float delta = d_step == +1 ? rt_eps : -rt_eps;
          e.best.update(loss_chg, fid, e.last_fvalue + delta, d_step == -1);
        }
      }
    }
  };
 public:
  // set training parameter
  virtual void set_param(const char *name, const char *val) {
    param.set_param(name, val);
  }
  virtual void update(const std::vector<bst_gpair> &gpair,
                      IFMatrix *p_fmat,
                      const BoosterInfo &info,
                      const std::vector<RegTree*> &trees) {
    // rescale learning rate according to size of trees
    float lr = param.learning_rate;
    param.learning_rate = lr / trees.size();
    // build tree
    for (size_t i = 0; i < trees.size(); ++i) {
      Builder builder(param);
      builder.update(gpair, p_fmat, info, trees[i]);
    }
    param.learning_rate = lr;
  }
};
// \brief core statistics used for tree construction
struct GradStats {
  // \brief set leaf vector value based on statistics
  inline void SetLeafVec(const TrainParam &param, bst_float *vec) const{
  }
  // \brief sum gradient statistics
  double sum_grad;
  // \brief sum hessian statistics
  double sum_hess;
  // \brief add statistics to the data
  inline void add(double grad, double hess) {
    sum_grad += grad; sum_hess += hess;
  }
  // \brief accumulate statistics,
  // \param gpair the vector storing the gradient statistics
  // \param info the additional information 
  // \param ridx instance index of this instance
  inline void add(const std::vector<bst_gpair> &gpair,
                  const BoosterInfo &info,
                  bst_uint ridx) {
    const bst_gpair &b = gpair[ridx];
    this->add(b.grad, b.hess);
  }
  // \brief constructor, the object must be cleared during construction
  explicit GradStats(const TrainParam &param) {
    this->clear();
  }
  // \brief clear the statistics
  inline void clear(void) {
    sum_grad = sum_hess = 0.0f;
  }
  // \brief add statistics to the data
  inline void add(const GradStats &b) {
    this->add(b.sum_grad, b.sum_hess);
  }
  // \brief calculate gain of the solution
  inline double calc_gain(const TrainParam &param) const {
    return param.calc_gain(sum_grad, sum_hess);
  }
  // \brief caculate leaf weight
  inline double calc_weight(const TrainParam &param) const {
    return param.calc_weight(sum_grad, sum_hess);
  }
  // \return whether the statistics is not used yet
  inline bool empty(void) const {
    return sum_hess == 0.0;
  }
  // \brief set current value to a - b
  inline void set_substract(const GradStats &a, const GradStats &b) {
    sum_grad = a.sum_grad - b.sum_grad;
    sum_hess = a.sum_hess - b.sum_hess;
  }
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
