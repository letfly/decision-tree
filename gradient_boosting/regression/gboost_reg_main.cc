#include "regression/gboost_reg_train.h" // RegBoostTrain, train

int main(int argc, char *argv[]) {
  // Input
  char *config_path = argv[1];

  // Model build
  gboost::regression::RegBoostTrain train;

  // Output
  train.train(config_path, false);
  //gboost::regression::RegBoostTest test;
  //test.test(config_path, false);
  return 0;
}
