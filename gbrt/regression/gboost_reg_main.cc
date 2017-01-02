#include "gboost_reg_test.h"

int main(int argc, char **argv) {
  char *config_path = "demo/regression/reg.conf";
  //gboost::regression::RegBoostTrain train;
  gboost::regression::RegBoostTest test;
  //train.train(config_path, false);
  test.test(config_path, false);
}
