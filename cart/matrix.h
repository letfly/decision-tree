#ifndef CART_MATRIX_H
#define CART_MATRIX_H
#include <string>
#include <vector>

class Matrix {
 private:
  std::vector<std::vector<double> > elements;
  std::vector<std::string> column_labels;
  std::vector<std::string> row_labels;
 public:
  Matrix();
  void load(std::string filename, bool use_column_labels=true, bool use_row_lables=true);
  int columns();
  int rows();
  std::vector<double> column(int index);
  Matrix submatrix(std::vector<int> rows, std::vector<int> columns);
  void split(int column_index, double value, Matrix &m1, Matrix &m2);
  // Bracket overloaded operator:
  std::vector<double> &operator[](int i);
};
#endif
