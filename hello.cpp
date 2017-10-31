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
    for (int l = 0; l < layers; ++l) {
      data.push_back(vector<double>(st[l], 0.0));
    }
  }
  inline double &at(int layer, int i) {
    return data[layer][i];
  }
  void print() {
    cout << "Y:" << endl;
    for (int l = 0; l < layers; ++l) {
      cout << " ";
      for (int i = 0; i < nodes[l]; ++i) {
        cout << " " << at(l, i);
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
  for (int l = 1; l < layers; ++l) {
    for (int j = 0; j < nodes[l]; ++j) {
      double sum = 0.0;
      for (int i = 0; i < nodes[l - 1]; ++i) {
        sum += w.at(l - 1, i, j) * y.at(l - 1, i);
      }
      y.at(l, j) = sigmoid(sum);
    }
  }
}

vector<double> ApplyModel(W &w, const vector<double> &input) {
  Y y(w.nodes);
  FeedForward(w, y, input);
  vector<double> result;
  for (int i = 0; i < w.nodes[w.layers - 1]; ++i) {
    result.push_back(y.at(w.layers - 1, i));
  }
  return result;
}

void OutputDelta(const vector<double> &t, vector<double> &delta, Y &y) {
  delta.clear();
  const int lastLayer = y.layers - 1;
  for (size_t i = 0; i < t.size(); ++i) {
    double t_i = t[i];
    double y_i = y.at(lastLayer, i);
    delta.push_back((t_i - y_i) * y_i *(1.0 - y_i));
  }
}

void HiddenDelta(const vector<double> delta_next, vector<double> &delta, W &w, Y &y, int layer) {
  delta.clear();
  const int lastLayer = y.layers - 1;
  int n = y.nodes[layer];
  size_t m = delta_next.size();
  for (int j = 0; j < n; ++j) {
    double y_i = y.at(layer, j);
    double sum = 0.0;
    for (size_t i = 0; i < m; ++i) {
      sum += delta_next[i] * w.at(layer, j, i);
    }
    delta.push_back(sum * y_i * (1.0 - y_i));
  }
}

void BP(W &w, vector<Y> &ys, const vector<vector<double>> &outputs, const double alpha) {
  const int layers = ys[0].layers;
  const vector<int> &nodes = ys[0].nodes;
  const int patterns = ys.size();
  for (int p = 0; p < patterns; ++p) {
    Y &y = ys[p];
    const vector<double> &output = outputs[p];
    vector<double> delta;
    OutputDelta(output, delta, y);
    for (int layer = layers - 2; layer >= 0; --layer) {
      for (int k = 0; k < nodes[layer]; ++k) {
        for (int j = 0; j < nodes[layer + 1]; ++j) {
          double dw = alpha * delta[j] * y.at(layer, k);
          w.at(layer, k, j) = dw + w.at(layer, k, j);
        }
      }
      if (layer != 0) {
        HiddenDelta(delta, delta, w, y, layer);
      }
    }
  }
}

int main(int argc, char** argv) {
  vector<vector<double>> inputs;
  vector<vector<double>> outputs;
  for (int i = 0; i < 100; ++i) {
    vector<double> x;
    vector<double> y;
    x.push_back(i);
    y.push_back(i % 2);
    inputs.push_back(x);
    outputs.push_back(y);
  }
  vector<int> nodes{ 1, 2, 1 };
  W w(nodes);
  w.at(0, 0, 0) = 0.29;
  w.at(0, 0, 1) = 0.31;
  w.at(1, 0, 0) = 0.42;
  w.at(1, 1, 0) = 0.64;
  w.print();
  vector<Y> ys(inputs.size(), Y(nodes));
  for (int iter = 0; iter < 20; ++iter) {
    for (int p = 0; p < inputs.size(); ++p) {
      FeedForward(w, ys[p], inputs[p]);
    }
    BP(w, ys, outputs, 0.1);
    w.print();
  }
  cout << ApplyModel(w, vector<double>(1, 102.0))[0] << endl;
  cout << ApplyModel(w, vector<double>(1, 103.0))[0] << endl;
  return 0;
}
