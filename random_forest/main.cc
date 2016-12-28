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
double test(Classifier *c, Matrix &m, std::vector<int> &classes) {
  classes.empty();
  // Analyze the results of the tree against training dataset
  int right = 0;
  int wrong = 0;
  for (int i = 0; i < m.rows(); ++i) {
    std::vector<double> &row = m[i];
    int actual_class = row[row.size()-1];
    int predict_class = c->classify(row);
    printf("predict=%dactual=%d", predict_class, actual_class);
    classes.push_back(actual_class);
    if (predict_class == actual_class) ++right;
    else ++wrong;
  }
  double percent = right*100.0/m.rows();
  printf("percent=%f\n", percent);
  return percent;
}
void train_only(Matrix &matrix) {
  std::vector<Classifier*> classifiers;
  std::vector<int> classes;
  if (n_features > matrix.columns()-1 || n_features <= 0) n_features = matrix.columns()-1;
  classifiers.push_back(new ParallelForest(n_trees, n_features, n_threads));

  for (int i = 0; i < classifiers.size(); ++i) {
    Classifier *classifier = classifiers[i];
    printf("training classifier #%d\n", i);
    classifier->train(matrix); // Model build
    double percent = test(classifier, matrix, classes); // Output
    printf("training set recovered: %f%%\n", percent);
  }
}
double train_and_test(Matrix &train, Matrix &testing) {
  ParallelForest forest(n_trees, n_features, n_threads);
  forest.train(train);
  std::vector<int> classes;
  Classifier *classifier = &forest;
  double percent = test(classifier, testing, classes);
  printf("%f%% correct\n", percent);
  std::vector<double> class_doubles(classes.begin(), classes.end());
  testing.append_column(class_doubles, "Class");

  return percent;
}
void folded_train_and_test(Matrix &input_matrix, int n_folds, std::string &filename) {
  Matrix result;
  Matrix matrix = input_matrix.shuffled();
  int R = matrix.rows();
  int N = R/n_folds;
  std::vector<int> all_columns = range(0, matrix.columns());
  double total_percent = 0.0;
  for (int i = 0; i < n_folds; ++i) {
    printf("Training and Testing Fold #%d\n", i);
    ParallelForest forest(n_trees, n_features, n_threads);

    // Get training subset
    std::vector<int> training_rows;
    // Begining fold
    if (i == 0) training_rows = range(N, R);
    // Middle fold
    else if (i < n_folds-1) training_rows = merge(range(0, i*N), range((i+1)*N, R));
    // Last fold
    else training_rows = range(0, R-N);
    Matrix training = matrix.submatrix(training_rows, all_columns);
    // Get testing subset
    std::vector<int> testing_rows = range(i*N, (i+1)*N);
    // Include extra elements into last fold
    if (i == n_folds-1) testing_rows = range(i*N, R);
    Matrix testing = matrix.submatrix(testing_rows, all_columns);
    // Test
    total_percent += train_and_test(training, testing);
    printf("n_f=%dt_p=%f\n", i, total_percent);
    // Store results (classID is in testing)
    result.merge_rows(testing);
  }
  double percent = total_percent/n_folds;
  printf("Percent recovered: %f%%\n", percent);

  std::vector<int> rows = range(result.rows());
  std::vector<int> cols;
  cols.push_back(result.columns()-1);
  printf("%lu\t%lu\n", rows.size(), cols.size());
  printf("%d\t%d\n", result.rows(), result.columns());
  Matrix sub = result.submatrix(rows, cols);
  sub.save(filename.c_str());
}

int main(int argc, char *argv[]) {
  // Input
  char *cvalue = NULL;
  int c;
  std::string train_file, save_file;
  while ((c = getopt(argc, argv, "t:s:c:p:n:f:m:")) != -1) {
    switch(c) {
      case 't': train_file = optarg; break;// Train file
      case 's': save_file = optarg; break;// Train file
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
  m.load(train_file);
  printf("%d rows and %d columns", m.rows(), m.columns());

  // Model build and Output
  //train_only(m);
  folded_train_and_test(m, 10, save_file);

  return 0;
}
