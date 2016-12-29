#include <cassert>
#include "stats.h" // mode()
#include "tree_node.h" // identifier 'TreeNode'
#include "util.h" // join()

static double MINIMUM_GAIN = 0.001;

TreeNode::TreeNode() { // TreeNode::train() in tree_node.cc
  left = right = NULL;
  column = -1;
  value = 1337.1337;
  classification = -1;
}

TreeNode::~TreeNode() {
  if (left != NULL) delete left;
  if (right != NULL) delete right;
}

int TreeNode::count() { // root.count() in main.cc
  int result = 1;
  if (left != NULL) result += left->count();
  if (right != NULL) result += right->count();
  return result;
}

double regression_score(Matrix &matrix, int col_index) { // TreeNode::train() in tree_node.cc O(basic_linear_regression)
  std::vector<double> x = matrix.column(col_index); // Fill the data in the col_index column to x
  std::vector<double> y = matrix.column(-1); // Fill the last column category to y // y = {0.000000, 1.000000, 1.000000, 1.000000, 2.000000}
  //for (auto i: x) printf("i=%f ", i);
  double k, b;
  basic_linear_regression(x, y, k, b);
  double error = sum_of_squares(x, y, k, b);
  return error;
}

void TreeNode::train(Matrix &m, std::vector<int> columns) { // root.train() in main.cc
  //printf("training on %s\n", join(columns, ' ').c_str());
  // Edge cases;
  assert(m.rows() > 0); // If wrong, stop the programming
  assert(m.columns() > 0);
  if (columns.size() == 0) {
    classification = mode(m.column(-1));
    return ;
  }
  // Decide which column to split on
  double min_error = 1000000000.0;
  int min_index = columns[0];
  double error = min_error;
  for (int i = 0; i < columns.size(); ++i) {
    int column = columns[i];
    error = regression_score(m, column); // Calculate the linear regression for each feature
    //printf("error=%f\n", error);
    if (error < min_error) {
      min_index = column;
      min_error = error;
    }
  }
  // Split on lowest error-column
  double v = mean(m.column(min_index)); // Calculate the average
  Matrix l, r;
  m.split(min_index, v, l, r); // Take the min_index column, less than v fill to the left subtree, else fill to the right subtree
  if (l.rows()<=0 || r.rows()<=0) {
    //printf("l or r: 0 rows \n");
    classification = mode(m.column(-1)); // Predict
    return ;
  }
  double left_error = regression_score(l, min_index);
  double right_error = regression_score(r, min_index);
  //printf("m_e=%f,l=%f,r=%f,v=%f\n", min_error, left_error, right_error, v);
  double gain = min_error-(left_error-right_error);
  if (gain < MINIMUM_GAIN) {
    //printf("split on min gain: %f %f %f", left_error, right_error, gain);
    classification = mode(m.column(-1));
    return ;
  }
  column = min_index;
  value = v;
  // train child nodes in tree
  left = new TreeNode();
  left->train(l, columns);
  right = new TreeNode();
  right->train(r, columns);
  //printf("Splitting on column %d with value %f\n", min_index, value);
}

int TreeNode::classify(std::vector<double> &row) { // root.classify() in main.cc
  if (classification != -1) return classification;
  if (row[column] < value) return left->classify(row);
  else return right->classify(row);
}
