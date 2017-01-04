#ifndef CART_STATS_H_
#define CART_STATS_H_
#include <vector>

double mode(const std::vector<double> &list);
void basic_linear_regression(const std::vector<double> &x, const std::vector<double> &y, double &m, double &b);
double sum_of_squares(const std::vector<double> &x, const std::vector<double> &y, double m, double b);
double mean(const std::vector<double> &list);
#endif
