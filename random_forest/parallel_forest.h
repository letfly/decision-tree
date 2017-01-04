#ifndef PARALLEL_FOREST_H_
#define PARALLEL_FOREST_H_
#include "random_forest/forest.h" // Forest, Matrix, classify

class ParallelForest : public Forest {
 protected:
  int n_threads;
 public:
  ParallelForest();
  ParallelForest(int n_trees, int n_features, int n_threads);
  virtual void train(Matrix &m);
};
#endif
