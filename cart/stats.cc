#include <map>
#include <vector>
#include "stats.h"

double sum(const std::vector<double> &list) { // O(size)
  double result = 0.0;
  for (int i = 0; i < list.size(); ++i) result += list[i];
  return result;
}

double sum_squared(const std::vector<double> &list) { // O(list.size())
  double result = 0.0;
  for (int i = 0; i < list.size(); ++i) result += list[i]*list[i];
  return result;
}

double covariance(const std::vector<double> &dist1, const std::vector<double> &dist2) {
  double result = 0.0;
  for (int i = 0; i < dist1.size(); ++i) result += dist1[i]*dist2[i];
  return result;
}

double mode(const std::vector<double> &list) { // return the most nums in list
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
  //printf("max=%f \n", (*max).first);
  return (*max).first;
}

void basic_linear_regression(const std::vector<double> &x, const std::vector<double> &y, double &k, double &b) { // one_variance O(rows_size*rows_size)
  int length = x.size();
  double sum_x = sum(x);
  double sum_y = sum(y);

  double sum_x_squared = sum_squared(x); // 计算平方和
  double cov = covariance(x, y);

  double numerator = (cov - (sum_x*sum_y)/length);
  double denominator = ( sum_x_squared - ((sum_x*sum_x)/length) );
  if (denominator == 0.0) k = 0.0;
  else k = numerator/denominator;
  b = (sum_y - k*sum_x)/length;
}

double sum_of_squares(const std::vector<double> &x, const std::vector<double> &y, double k, double b) {
  double result = 0.0;
  for (int i = 0; i < x.size(); ++i) {
    double expected = k*x[i]+b;
    double actual = y[i];
    double difference = expected-actual;
    double squared = difference*difference;
    result += squared;
  }
  return result;
}

double mean(const std::vector<double> &list) { return sum(list)/list.size(); }
