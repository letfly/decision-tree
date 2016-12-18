#include <cassert>
#include "tree_node.h" // identifier 'TreeNode'
#include "stats.h" // mode()
#include "util.h" // join()

static double MINIMUM_GAIN = 0.1;

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

double regression_score(Matrix &matrix, int col_index) { // TreeNode::train() in tree_node.cc
  std::vector<double> x = matrix.column(col_index);
  std::vector<double> y = matrix.column(-1);
  double m, b;
  basic_linear_regression(x, y, m, b);
  double error = sum_of_squares(x, y, m, b);
  return error;
}

void TreeNode::train(Matrix &m, std::vector<int> columns) { // root.train() in main.cc
  printf("training on %s\n", join(columns, ' ').c_str());
  // Edge cases;
  assert(m.rows() > 0); // if wrong, stop the programming
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
    error = regression_score(m, column);
    if (error < min_error) {
      min_index = column;
      min_error = error;
    }
  }
  // Split on lowest error-column
  double v = mean(m.column(min_index));
  Matrix l, r;
  m.split(min_index, v, l, r);
  if (l.rows()<=0 || r.rows()<=0) {
    printf("l or r: 0 rows \n");
    classification = mode(m.column(-1));
    return ;
  }
  printf("%d, %d\n", l.rows(), r.rows());
  double left_error = regression_score(l, min_index);
  double right_error = regression_score(r, min_index);
  double gain = error- (left_error-right_error);
  if (gain < MINIMUM_GAIN) {
    printf("split on min gain: %f %f %f", left_error, right_error, gain);
    classification = mode(m.column(-1));
    return ;
  }
  column = min_index;
  value = v;
  // train child nodes in tree
  std::vector<int> new_columns = columns;
  remove(new_columns.begin(), new_columns.end(), min_index);
  left = new TreeNode();
  left->train(l, new_columns);
  right = new TreeNode();
  right->train(r, new_columns);
  printf("Splitton on column %d with value %f\n", min_index, value);
}

int TreeNode::classify(std::vector<double> &row) { // root.classify() in main.cc
  if (classification != -1) return classification;
  if (row[column] < value) return left->classify(row);
  else return right->classify(row);
}
