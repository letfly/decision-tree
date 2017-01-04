#ifndef FOREST_H_
#define FOREST_H_
#include "cart/tree_node.h" // Classifier, TreeNode, Matrix

class Forest : public Classifier {
 protected:
  int n_trees;
  int n_features;
  std::vector<TreeNode> trees;
 public:
  Forest();
  Forest(int n_trees, int n_features);
  void init(int n_trees, int n_features);
  virtual void train(Matrix &m);
  virtual int classify(std::vector<double> &row);
};
#endif
