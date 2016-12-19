#include <map>
#include <vector>
#include "stats.h"

double sum(const std::vector<double> &list) { // basic_linear_regression() and mean() in stats.cc // O(size)
  double result = 0.0;
  for (int i = 0; i < list.size(); ++i) result += list[i];
  return result;
}

double sum_squared(const std::vector<double> &list) { // basic_linear_regression() in stats.cc // O(list.size())
  double result = 0.0;
  for (int i = 0; i < list.size(); ++i) result += list[i]*list[i];
  return result;
}

double mean(const std::vector<double> &list) { return sum(list)/list.size(); } // TreeNode::train() in tree_node.cc

double mode(const std::vector<double> &list) { // TreeNode::train() in tree_node.cc
  std::map<double, int> repeats;
  for (int i =0; i < list.size(); ++i) {
    double value = list[i];
    if (repeats.find(value) == repeats.end()) repeats[value] = 1;
    else repeats[value] += 1;
  }
  // http://stackoverflow.com/questions/9370945/c-help-finding-the-max-value-in-a-map
  auto max = max_element(repeats.begin(), repeats.end(),
    [](const std::pair<double, int> &p1, const std::pair<double, int> &p2) { return p1.second<p2.second; }
  );
  return (*max).first;
}

double covariance(const std::vector<double> &dist1, const std::vector<double> &dist2) { // basic_linear_regression() in stats.cc
  double result = 0.0;
  for (int i = 0; i < dist1.size(); ++i) result += dist1[i]*dist2[i];
  return result;
}

void basic_linear_regression(const std::vector<double> &x, const std::vector<double> &y, double &m, double &b) { // test_regression() in stats.cc and regression_score() in tree_node.cc // 一元线性回归
  int length = x.size();
  double sum_x = sum(x);
  double sum_y = sum(y);

  double sum_x_squared = sum_squared(x); // 计算平方和
  double cov = covariance(x, y);

  double numerator = (cov - (sum_x*sum_y)/length);
  double denominator = ( sum_x_squared - ((sum_x*sum_x)/length) );
  if (denominator == 0.0) m = 0.0;
  else m = numerator/denominator;
  b = (sum_y - m*sum_x)/length;
}

void test_regression() { // test_regression() in main.cc
  std::vector<double> x;
  x.push_back(0.0);
  x.push_back(1.0);
  x.push_back(2.0);
  std::vector<double> y;
  y.push_back(3.0);
  y.push_back(5.0);
  y.push_back(8.0);
  double m, b;
  basic_linear_regression(x, y, m, b);
}

double sum_of_squares(const std::vector<double> &x, const std::vector<double> &y, double m, double b) { // regression_score() in tree_node.cc
  double result = 0.0;
  for (int i = 0; i < x.size(); ++i) {
    double expected = m*x[i]+b;
    double actual = y[i];
    double difference = expected-actual;
    double squared = difference*difference;
    result += squared;
  }
  return result;
}
