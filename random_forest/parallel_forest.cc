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
  printf("parallel forest training %lu\n", trees.size());

  // 创建线程池
  Pool *pool = pool_start(training_thread, n_threads);
  // 启动线程
  std::vector<std::vector<int> > all_subsets(trees.size());
  std::vector<int> all_columns = range(m.columns()-1);
  for (int i = 0; i < trees.size(); ++i) {
    TreeNode &tree = trees[i];
    random_shuffle(all_columns.begin(), all_columns.end());
    all_subsets[i] = slice(all_columns, 0, n_features);

    // 创建work
    Work *work = new struct Work;
    work->matrix = &m;
    work->tree = &tree;
    work->subset = &all_subsets[i];

    pool_enquence(pool, work, true);
  }
  // 加入所有
  pool_wait(pool);
  // 释放资源
  pool_end(pool);
}
