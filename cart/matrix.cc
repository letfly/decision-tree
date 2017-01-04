#include <cassert> // assert
#include <fstream> // ifstream
#include <string> // string
#include "cart/matrix.h"
#include "cart/util.h" // split_string, range, join

Matrix::Matrix() {}

void Matrix::load(std::string filename, bool use_column_labels, bool use_row_lables) { // O(elements.size()*elements[0].size())
  std::ifstream file(filename);
  std::string line;
  int line_number = 0;
  while (getline(file, line)) {
    std::vector<std::string> tokens = split_string(line, ",");
    if (tokens.size() == 0) {
      //printf("Matrix.load(): skipping blank line on line #%d\n", line_number);
      ++line_number;
      continue;
    }
    if (use_column_labels && line_number==0) { // Removes the first row of field names
      column_labels = tokens; // Fill in the first line of the field name to column_labels
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

int Matrix::rows() { return elements.size(); }// O(1)

int Matrix::columns() { // O(1)
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

/*
 * m = [[0,1],[2,3],[4,5]]
 * m.submatrix([0,1],[1]) = [[1],[3]]
 * */
Matrix Matrix::submatrix(std::vector<int> rows, std::vector<int> columns) { // O(rows_size*columns_size)
  Matrix m;
  for (int j = 0; j < columns.size(); ++j)
    m.column_labels.push_back(column_labels[j]);

  for (int i = 0; i < rows.size(); ++i) {
    int y = rows[i];
    std::vector<double> row;
    m.row_labels.push_back(row_labels[y]);
    for (int j = 0; j < columns.size(); ++j) {
      int x = columns[j];
      //printf("y=%dx=%delement=%f", y, x, elements[y][x]);
      row.push_back(elements[y][x]);
    }
    m.elements.push_back(row);
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
void Matrix::split(int column_index, double value, Matrix &m1, Matrix &m2) { // O(elements.size()*elements[0].size())
  std::vector<int> m1_rows;
  std::vector<int> m2_rows;
  for (int i = 0; i < elements.size(); ++i) {
    double element = elements[i][column_index];
    if (element < value) m1_rows.push_back(i);
    else m2_rows.push_back(i);
  }
  std::vector<int> all_cols = range(columns()); // {0, cols-1}
  //for (auto i: m1_rows) printf("i=%dmin_index=%d", i, column_index);
  m1 = submatrix(m1_rows, all_cols); // Take the m1_rows row, all_cols column of data: matrix[m1_rows, all_cols]
  m2 = submatrix(m2_rows, all_cols); // m2_rows={0,1,2}, all_cols={0} matrix[m2_rows, all_cols]
}

// This function returns a shuffled version of the Matrix
Matrix Matrix::shuffled() {
  std::vector<int> row_indices = range(rows());
  random_shuffle(row_indices.begin(), row_indices.end());
  std::vector<int> column_indices = range(columns());
  return submatrix(row_indices, column_indices);
}

// Add rows from other Matrix to this Matrix
void Matrix::merge_rows(Matrix &other) {
  if (columns() == 0) column_labels = other.column_labels;
  assert(columns()==other.columns() || rows()==0);
  for (int i = 0; i < other.rows(); ++i) {
    std::vector<double> &row = other.elements[i];
    elements.push_back(row);
    row_labels.push_back(other.row_labels[i]);
  }
}

// Append a column to the right side of the Matrix
void Matrix::append_column(std::vector<double> &col) {
  assert(col.size() == rows());
  for (int i = 0; i < col.size(); ++i) {
    double value = col[i];
    elements[i].push_back(value);
  }
}

void Matrix::save(std::string filename, std::string name) {
  std::ofstream file(filename.c_str());
  // Write column header
  column_labels.push_back(name);
  if (column_labels.size() > 0) file << join(column_labels, ',');
  file << '\n';
  // Write elements
  for (int i = 0; i < elements.size(); ++i) {
    std::vector<double> &row = elements[i];
    if (row_labels.size() > 0) file << row_labels[i] << ',';
    file << join(row, ',');
    file << '\n';
  }
  file.close();
}
