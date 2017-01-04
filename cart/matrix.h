#ifndef CART_MATRIX_H_
#define CART_MATRIX_H_
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
  int rows();
  int columns();
  std::vector<double> &operator[](int i);
  std::vector<double> column(int index);
  Matrix submatrix(std::vector<int> rows, std::vector<int> columns);
  void split(int column_index, double value, Matrix &m1, Matrix &m2);

  Matrix shuffled();
  void merge_rows(Matrix &other);
  void append_column(std::vector<double> &col);
  void save(std::string filename, std::string name="");
  // Bracket overloaded operator:
};
#endif
