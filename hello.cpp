#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

class W {
private:
  vector<vector<double>> data;
public:
  const int layers;
  const vector<int> nodes;
  W(vector<int> st) : layers(st.size()), nodes(st), data() {
    for (int i = 0; i < layers - 1; ++i) {
      data.push_back(vector<double>(st[i] * st[i + 1], 0.0));
    }
  }
  inline double &at(int n, int i, int j) {
    return data[n][i * nodes[n + 1] + j];
  }
  void print() {
    for (int n = 0; n < layers - 1; ++n) {
      cout << "W[" << n << "]:" << endl;
      for (int i = 0; i < nodes[n]; ++i) {
        cout << " ";
        for (int j = 0; j < nodes[n + 1]; ++j) {
          cout << " " << at(n, i, j);
        }
        cout << endl;
      }
    }
  }
};

class Y {
private:
  vector<vector<double>> data;
public:
  const int layers;
  const vector<int> nodes;
  Y(vector<int> st) : layers(st.size()), nodes(st), data() {
    for (int i = 0; i < layers; ++i) {
      data.push_back(vector<double>(st[i], 0.0));
    }
  }
  inline double &at(int n, int i) {
    return data[n][i];
  }
  void print() {
    cout << "Y:" << endl;
    for (int n = 0; n < layers; ++n) {
      cout << " ";
      for (int i = 0; i < nodes[n]; ++i) {
        cout << " " << at(n, i);
      }
      cout << endl;
    }
  }
};

double sigmoid(double z) {
  return 1.0 / (1.0 + std::exp(-z));
}

void FeedForward(W &w, Y &y, const vector<double> &input) {
  for (int i = 0; i < input.size(); ++i) {
    y.at(0, i) = input[i];
  }
  const int layers = y.layers;
  const vector<int> &nodes = y.nodes;
  for (int n = 1; n < layers; ++n) {
    for (int j = 0; j < nodes[n]; ++j) {
      double sum = 0.0;
      for (int i = 0; i < nodes[n - 1]; ++i) {
        sum += w.at(n - 1, i, j) * y.at(n - 1, i);
      }
      y.at(n, j) = sigmoid(sum);
    }
  }
}

void BP(W &w, Y &y, const vector<double> &output) {
  const int layers = y.layers;
  const vector<int> &nodes = y.nodes;

}

int main(int argc, char** argv) {
  vector<int> nodes{1, 2, 1};
  W w(nodes);
  Y y(nodes);
  w.at(0, 0, 0) = 0.29;
  w.at(0, 0, 1) = 0.31;
  w.at(1, 0, 0) = 0.42;
  w.at(1, 1, 0) = 0.64;
  w.print();
  y.print();
  const vector<double> input{ 1.0 };
  FeedForward(w, y, input);
  y.print();
  return 0;
}
