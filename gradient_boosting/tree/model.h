#ifndef TREE_MODEL_H_
#define TREE_MODEL_H_
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
    // pointer to left, right
    int cleft_, cright_;
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
  };
  // \brief model parameter
  Param param;
 public:
  // \brief get node given nid
  inline Node &operator[](int nid) {
    return nodes[nid];
  }
  // \brief get node given nid
  inline const Node &operator[](int nid) const {
    return nodes[nid];
  }
  // \brief get leaf vector given nid
  inline bst_float* leafvec(int nid) {
    if (leaf_vector.size() == 0) return NULL;
    return &leaf_vector[nid * param.size_leaf_vector];
  }
};

// \brief node statistics used in regression tree
struct RTreeNodeStat {
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
