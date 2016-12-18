#ifndef CART_STATS_H
#define CART_STATS_H
#include <vector>

double sum(const std::vector<double> &list);
double sum_squared(const std::vector<double> &list);
double mean(const std::vector<double> &list);
double mode(const std::vector<double> &list);
double covariance(const std::vector<double> &dist1, const std::vector<double> &dist2);
void basic_linear_regression(const std::vector<double> &x, const std::vector<double> &y, double &m, double &b);
double sum_of_squares(const std::vector<double> &x, const std::vector<double> &y, double m, double b);
void test_regression();
#endif
