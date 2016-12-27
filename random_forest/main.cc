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
  // Analyze the results of the tree against training dataset
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
  if (n_features > matrix.columns()-1 || n_features <= 0) n_features = matrix.columns()-1;
  classifiers.push_back(new ParallelForest(n_trees, n_features, n_threads));

  for (int i = 0; i < classifiers.size(); ++i) {
    Classifier *classifier = classifiers[i];
    printf("training classifier #%d\n", i);
    classifier->train(matrix); // Model build
    double percent = test(classifier, matrix); // Output
    printf("training set recovered: %f%%\n", percent);
  }
}
int main(int argc, char *argv[]) {
  // Input
  char *cvalue = NULL;
  int c;
  std::string filename;
  while ((c = getopt(argc, argv, "t:c:p:n:f:m:")) != -1) {
    switch(c) {
      case 't': filename = optarg; break;// Train file
      case 'c': break; // Pridict category
      case 'p': n_threads = atoi(optarg); break;
      case 'n': n_trees = atoi(optarg); // The nums of threads
                assert(n_trees > 0); break;
      case 'f': n_features = atoi(optarg); // The nums of features selected
                assert(n_features > 0); break;
      case 'm': break; // The MINIMUM_GAIN
      default: exit(1);
    }
  }
  if (n_threads <= 0) n_threads = 16;
  Matrix m;
  m.load(filename);
  printf("%d rows and %d columns", m.rows(), m.columns());

  // Model build and Output
  train_and_test(m);

  return 0;
}
