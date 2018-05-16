#include <cmath>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

using namespace std;

void DrawLine(int len) {
  cout << setfill('-') << setw(len) << '-' << endl;
  cout << setfill(' ') << setw(0);
}

void PrintV(const vector<double> &v) {
  for (const double &ele : v) {
    cout << setw(12) << left << fixed << setprecision(5) << ele;
  }
  cout << endl;
  cout << setw(0);
}

void PrintW(const vector<vector<double>> &w) {
  for (const vector<double> &v : w) {
    PrintV(v);
  }
}

vector<double> Multiply(const vector<double> &v, const vector<vector<double>> &w) {
  vector<double> result;
  const auto m = v.size();
  const auto n = w[0].size();
  result.reserve(n);
  for (int j = 0; j < n; ++j) {
    result.push_back(v[0] * w[0][j]);
  }
  for (int i = 1; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      result[j] += v[i] * w[i][j];
    }
  }
  return result;
}

vector<double> Softmax(const vector<double> &v) {
  vector<double> result(v.size());
  double sum{0.0};
  for (int i = 0; i < v.size(); ++i) {
    result[i] = exp(v[i]);
    sum += result[i];
  }
  for (double &x : result) {
    x /= sum;
  }
  return result;
}

int MaxIndex(const vector<double> &v) {
  int res{0};
  double max{v[0]};
  for (int i = 1; i < v.size(); ++i) {
    if (v[i] > max) {
      res = i;
      max = v[i];
    }
  }
  return res;
}

int main(int argc, char **argv) {
  const int vocab_size = 9;
  const int hidden_layer_size = 3;
  // initialize w
  vector<vector<vector<double>>> w;
  for (int i = 0; i < 2; ++i) {
    w.emplace_back(vector<vector<double>>{});
  }
  for (int i = 0; i < vocab_size; ++i) {
    w[0].emplace_back(vector<double>(static_cast<unsigned long>(hidden_layer_size), 0.0));
  }
  for (int i = 0; i < hidden_layer_size; ++i) {
    w[1].emplace_back(vector<double>(static_cast<unsigned long>(vocab_size), 0.0));
  }
  // randomize w
  std::default_random_engine generator{11};
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  for (auto &e : w) {
    for (std::vector<double> &v : e) {
      for (double &ele : v) {
        ele = distribution(generator);
      }
    }
  }
  // debug w
  for (auto &e : w) {
    DrawLine(80);
    PrintW(e);
  }
  DrawLine(80);
  PrintV(w[0][0]);
  DrawLine(80);
  auto output = Multiply(w[0][1], w[1]);
  PrintV(output);
  DrawLine(80);
  output = Softmax(output);
  PrintV(output);
  cout << "Max Index: " << MaxIndex(output) << endl;
  return 0;
}