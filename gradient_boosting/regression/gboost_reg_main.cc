#include "regression/gboost_reg_train.h"

int main(int argc, char **argv) {
  // Input
  char *config_path = "demo/regression/reg.conf";
  printf("dd");

  // Model build
  gboost::regression::RegBoostTrain train;

  // Output
  train.train(config_path, false);
  //gboost::regression::RegBoostTest test;
  //test.test(config_path, false);
  return 0;
}
