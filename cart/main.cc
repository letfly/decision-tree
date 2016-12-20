#include <cstdio>
#include "stats.h"
#include "tree_node.h"
#include "matrix.h"
#include "util.h"
#include "forest.h"
#include "parallel_forest.h"

double test(Classifier *c, Matrix &m) {
  // 分析
  int right = 0;
  int wrong = 0;
  for (int i = 0; i < m.rows(); ++i) {
    std::vector<double> &row = m[i];
    int predict_class = row[row.size()-1];
    int actual_class = c->classify(row);
    if (predict_class == actual_class) ++right;
    else ++wrong;
  }
  double percent = right*100.0/m.rows();
  return percent;
}
void train_and_test(Matrix &matrix, int tree_size) {
  std::vector<Classifier*> classifiers;
  classifiers.push_back(new ParallelForest(tree_size, matrix.columns()-1, 4));

  for (int i = 0; i < classifiers.size(); ++i) {
    Classifier *classifier = classifiers[i];
    printf("training classifier #%d\n", i);
    classifier->train(matrix);
    double percent = test(classifier, matrix);
    printf("training set recovered: %f%%\n", percent);
  }
}
int main(int argc, char *argv[]) {
  // 输入
  std::string filename(argv[1]);
  std::string output_filename(argv[2]);
  int tree_size = 3;
  Matrix m;
  m.load(filename);
  printf("%d rows and %d columns", m.rows(), m.columns());

  // 模型建立
  TreeNode root;
  std::vector<int> columns = range(m.columns()-1); // 训练的列数
  root.train(m, columns);
  printf("%d nodes in tree\n", root.count());
  Forest f(tree_size, m.columns()-1);
  f.train(m);
  // 模型验证

  // 输出
  train_and_test(m, tree_size);

  return 0;
}
