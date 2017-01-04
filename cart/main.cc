#include <cstdio>
#include "cart/matrix.h" // Matrix, load, rows, columns, operator
#include "cart/tree_node.h" // TreeNode, train, count, classify
#include "cart/util.h" // range

int main(int argc, char *argv[]) {
  // Input
  std::string filename(argv[1]);
  std::string output_filename(argv[2]);
  Matrix m;
  m.load(filename);
  printf("\n\n%d rows and %d columns\n", m.rows(), m.columns());

  // Model build
  TreeNode tree;
  std::vector<int> columns = range(2); // the columns of features
  tree.train(m, columns);
  printf("%d nodes in tree\n", tree.count());

  // Output:Model validation
  // Analyze the results of the tree against training dataset
  int right = 0;
  int wrong = 0;
  for (int i = 0; i < m.rows(); ++i) {
    std::vector<double> &row = m[i];
    int actual_class = tree.classify(row);
    int expected_class = row[row.size()-1];
    if (actual_class == expected_class) ++right;
    else ++wrong;
  }
  // Evaluate results against original training set
  double percent = right*100.0/m.rows();
  printf("train set correct: %f%%\n", percent);

  return 0;
}
