#ifndef CART_TREE_NODE_H_
#define CART_TREE_NODE_H_
#include <string>
#include "cart/classifier.h"

class TreeNode : public Classifier{
 private:
  TreeNode *left;
  TreeNode *right;
  int column;
  double value;
  int classification;
 public:
  TreeNode();
  ~TreeNode();
  void train(Matrix &m, std::vector<int> columns);
  int count();
  virtual int classify(std::vector<double> &row);
};
#endif
