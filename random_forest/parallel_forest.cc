#include <cstdio>
#include "parallel_forest.h"
#include "pthread_pool.h"
#include "util.h"

ParallelForest::ParallelForest() {
  init(2, 10);
  n_threads = 4;
}

ParallelForest::ParallelForest(int n_trees, int n_features, int n_threads) {
  this->n_threads = n_threads;
  init(n_trees, n_features);
}

struct Work {
  Matrix *matrix;
  TreeNode *tree;
  std::vector<int> *subset;
};

void *training_thread(void *void_ptr) {
  printf("training_thread...");
  Work *work = (Work*)void_ptr;
  work->tree->train(*work->matrix, *work->subset);
  return NULL;
}

void ParallelForest::train(Matrix &m) {
  printf("parallel forest training with %lu trees and %d threads\n", trees.size(), n_threads);
  // Create thread pool
  void *pool = pool_start(&training_thread, n_threads);
  // Run through threads
  std::vector<std::vector<int> > all_subsets(trees.size());
  std::vector<int> all_columns = range(m.columns()-1);
  for (int i = 0; i < trees.size(); ++i) {
    TreeNode &tree = trees[i];
    random_shuffle(all_columns.begin(), all_columns.end());
    all_subsets[i] = slice(all_columns, 0, n_features);

    // Create work
    Work *work = new Work;
    work->matrix = &m;
    work->tree = &tree;
    work->subset = &all_subsets[i];

    pool_enqueue(pool, work, true);
  }
  // Join on all
  pool_wait(pool);
  // Free resources
  pool_end(pool);
}
