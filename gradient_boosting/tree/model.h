#ifndef TREE_MODEL_H_
#define TREE_MODEL_H_
#include <cstdlib>
#include <vector>
#include "data.h"
#include "utils/utils.h"

namespace gboost {
namespace tree {
// \brief template class of TreeModel 
// \tparam TSplitCond data type to indicate split condition
// \tparam TNodeStat auxiliary statistics of node to help tree building
template<typename TSplitCond, typename TNodeStat>
class TreeModel {
 private:
  class Node {
   private:
    // split feature index, left split or right split depends on the highest bit
    unsigned sindex_;
    // \brief in leaf node, we have weights, in non-leaf nodes, 
    //        we have split condition 
    union Info{
      float leaf_value;
      TSplitCond split_cond;
    };
    // extra info
    Info info_;
   public:
    // \brief get split condition of the node
    inline TSplitCond split_cond(void) const {
      return (this->info_).split_cond;
    }
    // pointer to left, right
    int cleft_, cright_;
    // \brief index of left child
    inline int cleft(void) const {
      return this->cleft_;
    }
    // \brief index of right child
    inline int cright(void) const {
      return this->cright_;
    }
    // \brief when feature is unknown, whether goes to left child
    inline bool default_left(void) const {
      return (sindex_ >> 31) != 0;
    }
    // \brief index of default child when feature is missing
    inline int cdefault(void) const {
      return this->default_left() ? this->cleft() : this->cright();
    }
    // \brief whether current node is leaf node
    inline bool is_leaf(void) const {
      return cleft_ == -1;
    }
    // \brief feature index of split condition
    inline unsigned split_index(void) const {
      return sindex_ & ((1U << 31) - 1U);
    }
    // \brief get leaf value of leaf node
    inline float leaf_value(void) const {
      return (this->info_).leaf_value;
    }
    // \brief set the leaf value of the node
    // \param value leaf value
    // \param right right index, could be used to store 
    //        additional information
    inline void set_leaf(float value, int right = -1) {
      (this->info_).leaf_value = value;
      this->cleft_ = -1;
      this->cright_ = right;
    }
    // pointer to parent, highest bit is used to
    // indicate whether it's a left child or not
    int parent_;
    // set parent
    inline void set_parent(int pidx, bool is_left_child = true) {
      if (is_left_child) pidx |= (1U << 31);
      this->parent_ = pidx;
    }
    // \brief get parent of the node
    inline int parent(void) const {
      return parent_ & ((1U << 31) - 1);
    }
    // \brief whether current node is root
    inline bool is_root(void) const {
      return parent_ == -1;
    }
    // \brief whether current node is left child
    inline bool is_left_child(void) const {
      return (parent_ & (1U << 31)) != 0;
    }
    // \brief set split condition of current node 
    // \param split_index feature index to split
    // \param split_cond  split condition
    // \param default_left the default direction when feature is unknown
    inline void set_split(unsigned split_index, TSplitCond split_cond,
                          bool default_left = false) {
      if (default_left) split_index |= (1U << 31);
      this->sindex_ = split_index;
      (this->info_).split_cond = split_cond;
    }
  };
  // vector of nodes
  std::vector<Node> nodes;
  // leaf vector, that is used to store additional information
  std::vector<bst_float> leaf_vector;

  // \brief parameters of the tree
  struct Param {
    // \brief leaf vector size, used for vector tree
    // used to store more than one dimensional information in tree
    int size_leaf_vector;
    // \brief number of start root
    int num_roots;
    // \brief  number of features used for tree construction
    int num_feature;
    // \brief set parameters from outside 
    // \param name name of the parameter
    // \param val  value of the parameter
    inline void set_param(const char *name, const char *val) {
      if (!strcmp("num_roots", name)) num_roots = atoi(val);
      if (!strcmp("num_feature", name)) num_feature = atoi(val);
      if (!strcmp("size_leaf_vector", name)) size_leaf_vector = atoi(val);
    }
    // \brief total number of nodes
    int num_nodes;
    // \brief number of deleted nodes
    int num_deleted;
  };
  // stats of nodes
  std::vector<TNodeStat> stats;
  // allocate a new node,
  // !!!!!! NOTE: may cause BUG here, nodes.resize
  inline int alloc_node(void) {
    if (param.num_deleted != 0) {
      int nd = deleted_nodes.back();
      deleted_nodes.pop_back();
      --param.num_deleted;
      return nd;
    }
    int nd = param.num_nodes++;
    utils::check(param.num_nodes < std::numeric_limits<int>::max(),
                 "number of nodes in the tree exceed 2^31");
    nodes.resize(param.num_nodes);
    stats.resize(param.num_nodes);
    leaf_vector.resize(param.num_nodes * param.size_leaf_vector); 
    return nd;
  }
 public:
  TreeModel(void) {
    param.num_nodes = 1;
    param.num_roots = 1;
  }
  // \brief get node given nid
  inline Node &operator[](int nid) {
    return nodes[nid];
  }
  // \brief get node given nid
  inline const Node &operator[](int nid) const {
    return nodes[nid];
  }
  // \brief get node statistics given nid
  inline TNodeStat &stat(int nid) {
    return stats[nid];
  }
  // \brief get leaf vector given nid
  inline bst_float* leafvec(int nid) {
    if (leaf_vector.size() == 0) return NULL;
    return &leaf_vector[nid * param.size_leaf_vector];
  }
  // \brief model parameter
  Param param;
  // \brief initialize the model
  inline void init_model(void) {
    param.num_nodes = param.num_roots;
    nodes.resize(param.num_nodes);
    stats.resize(param.num_nodes);
    leaf_vector.resize(param.num_nodes * param.size_leaf_vector, 0.0f);
    for (int i = 0; i < param.num_nodes; i ++) {
      nodes[i].set_leaf(0.0f);
      nodes[i].set_parent(-1);
    }
  }
  // \brief get current depth
  // \param nid node id
  // \param pass_rchild whether right child is not counted in depth
  inline int get_depth(int nid, bool pass_rchild = false) const {
    int depth = 0;
    while (!nodes[nid].is_root()) {
      if (!pass_rchild || nodes[nid].is_left_child()) ++depth;
      nid = nodes[nid].parent();
    }
    return depth;
  }
  // \brief number of extra nodes besides the root
  inline int num_extra_nodes(void) const {
    return param.num_nodes - param.num_roots - param.num_deleted;
  }
  // \brief get maximum depth
  // \param nid node id
  inline int max_depth(int nid) const {
    if (nodes[nid].is_leaf()) return 0;
    return std::max(max_depth(nodes[nid].cleft())+1,
                    max_depth(nodes[nid].cright())+1);
  }
  // \brief get maximum depth
  inline int max_depth(void) {
    int maxd = 0;
    for (int i = 0; i < param.num_roots; ++i)
      maxd = std::max(maxd, max_depth(i));
    return maxd;
  }
  // free node space, used during training process
  std::vector<int> deleted_nodes;
  // delete a tree node
  inline void delete_node(int nid) {
    utils::assert(nid >= param.num_roots, "can not delete root");
    deleted_nodes.push_back(nid);
    nodes[nid].set_parent(-1);
    ++param.num_deleted;
  }
  // \brief change a non leaf node to a leaf node, delete its children
  // \param rid node id of the node
  // \param new leaf value
  inline void change_to_leaf(int rid, float value) {
    utils::assert(nodes[nodes[rid].cleft() ].is_leaf(),
                  "can not delete a non termial child");
    utils::assert(nodes[nodes[rid].cright()].is_leaf(),
                  "can not delete a non termial child");
    this->delete_node(nodes[rid].cleft());
    this->delete_node(nodes[rid].cright());
    nodes[rid].set_leaf(value);
  }
  // \brief add child nodes to node
  // \param nid node id to add childs
  inline void add_childs(int nid) {
    int pleft  = this->alloc_node();
    int pright = this->alloc_node();
    nodes[nid].cleft_  = pleft;
    nodes[nid].cright_ = pright;
    nodes[nodes[nid].cleft() ].set_parent(nid, true);
    nodes[nodes[nid].cright()].set_parent(nid, false);
  }
};

// \brief node statistics used in regression tree
struct RTreeNodeStat {
  // \brief number of child that is leaf node known up to now
  int leaf_child_cnt;
  // \brief weight of current node
  float base_weight;
  // \brief loss chg caused by current split
  float loss_chg;
  // \brief sum of hessian values, used to measure coverage of data
  float sum_hess;
};

class RegTree: public TreeModel<bst_float, RTreeNodeStat> {
 public:
  // \brief dense feature vector that can be taken by RegTree
  // to do tranverse efficiently
  // and can be construct from sparse feature vector
  struct FVec {
    // \brief a union value of value and flag
    // when flag == -1, this indicate the value is missing
    union Entry{
      float fvalue;
      int flag;
    };
    std::vector<Entry> data;
    // \brief fill the vector with sparse vector
    inline void fill(const RowBatch::Inst &inst) {
      for (bst_uint i = 0; i < inst.length; ++i)
        data[inst[i].index].fvalue = inst[i].fvalue;
    }
    // \brief get ith value
    inline float fvalue(size_t i) const {
      return data[i].fvalue;
    }
    // \brief check whether i-th entry is missing
    inline bool is_missing(size_t i) const {
      return data[i].flag == -1;
    }
    // \brief drop the trace after fill, must be called after fill
    inline void drop(const RowBatch::Inst &inst) {      
      for (bst_uint i = 0; i < inst.length; ++i) {
        data[inst[i].index].flag = -1;
      }
    }
    // \brief intialize the vector with size vector
    inline void init(size_t size) {
      Entry e; e.flag = -1;
      data.resize(size);
      std::fill(data.begin(), data.end(), e);
    }
  };
  // \brief get next position of the tree given current pid
  inline int get_next(int pid, float fvalue, bool is_unknown) const {
    float split_value = (*this)[pid].split_cond();
    if (is_unknown) {
      return (*this)[pid].cdefault();
    } else {
      if (fvalue < split_value) {
        return (*this)[pid].cleft();
      } else {
        return (*this)[pid].cright();
      }
    }
  }
  // \brief get the leaf index 
  // \param feats dense feature vector, if the feature is missing the field is set to NaN
  // \param root_gid starting root index of the instance
  // \return the leaf index of the given feature 
  inline int get_leaf_index(const FVec&feat, unsigned root_id = 0) const {
    // start from groups that belongs to current data
    int pid = static_cast<int> (root_id);
    // tranverse tree
    while (!(*this)[pid].is_leaf()) {
      unsigned split_index = (*this)[pid].split_index();
      pid = this->get_next(pid, feat.fvalue(split_index), feat.is_missing(split_index));
    }
    return pid;
  }
};

}
}
#endif
