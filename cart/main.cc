#include <cstdio>
#include "stats.h"
#include "tree_node.h"
#include "matrix.h"
#include "util.h"

int main(int argc, char *argv[]) {
  // 输入
  std::string filename(argv[1]);
  std::string output_filename(argv[2]);
  Matrix m;
  m.load(filename);

  // Matrix testing
  Matrix m1, m2;
  m.split(1, 0.001, m1, m2);
  printf("%d\trows2=%d\n", m1.rows(), m2.rows());
  std::vector<int> r = range(10);
  std::vector<int> c = range(10);
  Matrix s = m.submatrix(r, c);
  // Regression Tree
  TreeNode root;
  std::vector<int> columns = range(10);
  root.train(m, columns);
  printf("%d nodes in tree\n", root.count());

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

  // stats
  test_regression();
  std::vector<double> test_mode;
  test_mode.push_back(1.0);
  test_mode.push_back(2.0);
  test_mode.push_back(5.0);
  test_mode.push_back(2.0);
  test_mode.push_back(7.0);
  printf("%f\n", mode(test_mode));

  return 0;
}
