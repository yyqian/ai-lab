#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>

using namespace std;

class W {
private:
  vector<vector<double>> data;
public:
  const int layers;
  const vector<int> nodes;
  W(const vector<int> &st) : layers(st.size()), nodes(st), data() {
    for (int i = 0; i < layers - 1; ++i) {
      data.push_back(vector<double>(st[i] * st[i + 1], 0.0));
    }
    randomize();
  }
  double &at(int l, int i, int j) {
    return data[l][i * nodes[l + 1] + j];
  }
  void randomize() {
    for (vector<double> &v : data) {
      for (double &ele : v) {
        ele = static_cast<double>(rand()) / RAND_MAX;
      }
    }
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
  Y(const vector<int> &st) : layers(st.size()), nodes(st), data() {
    for (int l = 0; l < layers; ++l) {
      data.push_back(vector<double>(st[l], 0.0));
    }
  }
  vector<double> &at(int layer) {
    return data[layer];
  }
  double &at(int layer, int i) {
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

class Patterns {
private:
  vector<vector<double>> inputs;
  vector<vector<double>> outputs;
public:
  int size;
  Patterns() : size(0), inputs(), outputs() {
  }
  void add(const vector<double> &&input, const vector<double> &&output) {
    inputs.emplace_back(input);
    outputs.emplace_back(output);
    ++size;
  }
  vector<vector<double>> &getInputs() {
    return inputs;
  }
  vector<vector<double>> &getOutputs() {
    return outputs;
  }
};

// f(z) = 1 / [1 + exp(-z)]
inline double sigmoid(double z) {
  return 1.0 / (1.0 + std::exp(-z));
}

// df(z)/dz = f(z)[1 - f(z)]
inline double sigmoid_prime(double f) {
  return f * (1.0 - f);
}

void FeedForward(W &w, Y &y, const vector<double> &input) {
  // the first layer of Y is the input
  y.at(0).assign(input.begin(), input.end());
  const int layers = y.layers;
  const vector<int> &nodes = y.nodes;
  for (int l = 1; l < layers; ++l) {
    for (int j = 0; j < nodes[l]; ++j) {
      double sum = 0.0;
      for (int i = 0; i < nodes[l - 1]; ++i) {
        sum += w.at(l - 1, i, j) * y.at(l - 1, i);
      }
      // sigmoid is the activation function
      y.at(l, j) = sigmoid(sum);
    }
  }
}

vector<double> Apply(W &w, const vector<double> &input) {
  Y y(w.nodes);
  FeedForward(w, y, input);
  return y.at(w.layers - 1);
}

void OutputDelta(const vector<double> &o, vector<double> &delta, Y &y) {
  delta.clear();
  const int lastLayer = y.layers - 1;
  for (size_t i = 0; i < o.size(); ++i) {
    double o_i = o[i];
    double y_i = y.at(lastLayer, i);
    delta.push_back((o_i - y_i) * sigmoid_prime(y_i));
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
    delta.push_back(sum * sigmoid_prime(y_i));
  }
}

void Backpropagation(W &w, vector<Y> &ys, const vector<vector<double>> &outputs, const double alpha) {
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

string join(vector<double> vec) {
  ostringstream ss;
  for (double v : vec) {
    ss << ' ' << v;
  }
  return ss.str();
}

int main(int argc, char** argv) {
  Patterns patterns;
  // XOR
  patterns.add(vector<double>{1, 1}, vector<double>{0});
  patterns.add(vector<double>{1, 0}, vector<double>{1});
  patterns.add(vector<double>{0, 1}, vector<double>{1});
  patterns.add(vector<double>{0, 0}, vector<double>{0});
  const double alpha = 1.0;
  const int iters = 100000;
  const vector<int> nodes{ 2, 5, 1 };
  W w(nodes);
  w.print();
  vector<Y> ys(patterns.size, Y(nodes));
  for (int iter = 0; iter < iters; ++iter) {
    for (int p = 0; p < patterns.size; ++p) {
      FeedForward(w, ys[p], patterns.getInputs()[p]);
    }
    Backpropagation(w, ys, patterns.getOutputs(), alpha);
    if (iter % (iters / 10) == 0) {
      cout << endl;
      cout << "====== iter " << iter << " ======" << endl;
      w.print();
      for (vector<double> x : patterns.getInputs()) {
        cout << join(x) << ": " << Apply(w, x)[0] << endl;
      }
    }
  }
  return 0;
}
