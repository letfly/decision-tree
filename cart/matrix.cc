#include <cassert> // assert
#include <fstream> // ifstream
#include <string> // string
#include "matrix.h"
#include "util.h"

Matrix::Matrix() {}

void Matrix::load(std::string filename, bool use_column_labels, bool use_row_lables) { // m.load(filename) in main.cc // O(elements.size()*elements[0].size())
  std::ifstream file(filename);
  std::string line;
  int line_number = 0;
  while (getline(file, line)) {
    std::vector<std::string> tokens = split_string(line, ",");
    if (tokens.size() == 0) {
      printf("Matrix.load(): skipping blank line on line #%d\n", line_number);
      ++line_number;
      continue;
    }
    if (use_column_labels && line_number==0) { // 移除第一行字段名
      column_labels = tokens; // 填充第一行字段名至列表签 column_labels
      ++line_number;
      continue;
    } else {
      if (use_row_lables) {
        row_labels.push_back(tokens[0]); // 填充第一列id至行标签 row_labels
        tokens.erase(tokens.begin() + 0);
      }
      std::vector<double> row;
      for (int i = 0; i < tokens.size(); ++i) { // 填充某一行除了第一个标签至row
        double element = stod(tokens[i]);
        row.push_back(element);
      }
      elements.push_back(row); // 填充除了第一行,第一列标签至elements
      ++line_number;
    }
  }
  file.close();
}

int Matrix::rows() { return elements.size(); } // TreeNode::train m.rows() in tree_node.cc // O(1)

/*
 * m = [[0,1],[2,3],[4,5]]
 * m.submatrix([0,1],[1]) = [[1],[3]]
 * */
Matrix Matrix::submatrix(std::vector<int> rows, std::vector<int> columns) { // m.submatrix(r, c) in main.cc and split() in matrix.cc // O(rows_size*columns_size)
  Matrix m;
  for (int j = 0; j < columns.size(); ++j) {
    m.column_labels.push_back(column_labels[j]);
  }
  for (int i = 0; i < rows.size(); ++i) {
    int y = rows[i];
    std::vector<double> row;
    m.row_labels.push_back(row_labels[y]);
    for (int j = 0; j < columns.size(); ++j) {
      int x = columns[j];
      row.push_back(elements[y][x]);
    }
    m.elements.push_back(elements[i]);
  }
  return m;
}

/*
 * m = [[0,1],[0,2],[0,3]]
 * Matrix m1, m2;
 * m.split(1,2,m1,m2);
 * m1 = [[0,1]]
 * m2 = [[0,2],[0,3]]
 * */
void Matrix::split(int column_index, double value, Matrix &m1, Matrix &m2) { // m.split() in main.cc // O(elements.size()*elements[0].size())
  std::vector<int> m1_rows;
  std::vector<int> m2_rows;
  for (int i = 0; i < elements.size(); ++i) {
    double element = elements[i][column_index];
    if (element < value) m1_rows.push_back(i);
    else m2_rows.push_back(i);
  }
  std::vector<int> all_cols = range(columns()); // {0, 列数-1}
  m1 = submatrix(m1_rows, all_cols); // 获取m1_rows行,all_cols列的数据 matrix[m1_rows, all_cols]
  m2 = submatrix(m2_rows, all_cols); // m2_rows={0,1,2}, all_cols={0} matrix[m2_rows, all_cols]
}

int Matrix::columns() { // TreeNode::train m.columns() in tree_node.cc // O(1)
  if (elements.size() == 0) return 0;
  else return elements[0].size();
}

std::vector<double> &Matrix::operator[](int i) { // std::vector<double> $row = m[i] in main.cc
  assert(i < elements.size());
  return elements[i];
}

std::vector<double> Matrix::column(int index) { // regression_score() matrix.column(col_index) in tree_node.cc // O(elements.size())
  std::vector<double> result;
  if (index < 0) index += columns();
  for (int i = 0; i < rows(); ++i) {
    double element = elements[i][index];
    result.push_back(element);
  }
  return result;
}
