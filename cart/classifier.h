#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include <cstdio>
#include "matrix.h"

class Classifier {
 public:
  virtual void train(Matrix &m) { printf("classifier no training"); };
  virtual int classify(std::vector<double> &row) { return 0; };
};
#endif
