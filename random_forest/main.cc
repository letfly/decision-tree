#include <cassert> // assert
#include <cstdio>
#include <unistd.h> // getopt, optarg
#include "matrix.h"
#include "stats.h"
#include "tree_node.h"
#include "util.h"
#include "forest.h"
#include "parallel_forest.h"

int n_trees, n_threads, n_features;
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
void train_and_test(Matrix &matrix) {
  std::vector<Classifier*> classifiers;
  if (n_features > matrix.columns()-1) n_features = matrix.columns()-1;
  classifiers.push_back(new ParallelForest(n_trees, n_features, n_threads));

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
  char *cvalue = NULL;
  int c;
  std::string filename;
  while ((c = getopt(argc, argv, "t:c:p:n:f:m:")) != -1) {
    switch(c) {
      case 't': filename = optarg; break;// 训练文件 
      case 'c': break; // 预测类别
      case 'p': n_threads = atoi(optarg); break;
      case 'n': n_trees = atoi(optarg); // 线程数
                assert(n_trees > 0); break;
      case 'f': n_features = atoi(optarg);
                assert(n_features > 0); break;
      case 'm': break; // 最小增益
      default: exit(1);
    }
  }
  if (n_threads <= 0) n_threads = 16;
  printf("%d ", n_threads);
  int n_trees = 3;
  Matrix m;
  m.load(filename);
  printf("%d rows and %d columns", m.rows(), m.columns());

  // 模型建立
  TreeNode root;
  std::vector<int> columns = range(m.columns()-1); // 训练的列数
  root.train(m, columns);
  printf("%d nodes in tree\n", root.count());
  Forest f(n_trees, m.columns()-1);
  f.train(m);
  // 模型验证

  // 输出
  train_and_test(m);

  return 0;
}
