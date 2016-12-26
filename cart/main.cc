#include <cstdio>
#include "matrix.h"
#include "stats.h"
#include "tree_node.h"
#include "util.h"

int main(int argc, char *argv[]) {
  // Input
  std::string filename(argv[1]);
  std::string output_filename(argv[2]);
  Matrix m;
  m.load(filename);

  // Model build
  TreeNode root;
  std::vector<int> columns = range(2); // the columns of features
  root.train(m, columns);
  printf("%d nodes in tree\n", root.count());

  // Output:Model validation
  // Analyze the results of the tree against training dataset
  int right = 0;
  int wrong = 0;
  for (int i = 0; i < m.rows(); ++i) {
    std::vector<double> &row = m[i];
    int actual_class = root.classify(row);
    int expected_class = row[row.size()-1];
    if (actual_class == expected_class) ++right;
    else ++wrong;
  }
  // Evaluate results against original training set
  double percent = right*100.0/m.rows();
  printf("training set recovered: %f%%\n", percent);

  return 0;
}
