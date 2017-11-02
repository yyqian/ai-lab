#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>

using namespace std;

class W {
private:
  vector<vector<double>> data;
  const int layers;
  const vector<int> neurons;
public:
  W(const vector<int> &neurons) : layers(neurons.size()), neurons(neurons), data() {
    for (int i = 0; i < layers - 1; ++i) {
      data.push_back(vector<double>(neurons[i] * neurons[i + 1], 0.0));
    }
    randomize();
  }
  double &at(int l, int i, int j) {
    return data[l][i * neurons[l + 1] + j];
  }
  const double &at(int l, int i, int j) const {
    return data[l][i * neurons[l + 1] + j];
  }
  const int getLayers() const {
    return layers;
  }
  const vector<int> &getNeurons() const {
    return neurons;
  }
  void randomize() {
    for (vector<double> &v : data) {
      for (double &ele : v) {
        ele = static_cast<double>(rand()) / RAND_MAX;
      }
    }
  }
  void print() const {
    for (int n = 0; n < layers - 1; ++n) {
      cout << "W[" << n << "]:" << endl;
      for (int i = 0; i < neurons[n]; ++i) {
        for (int j = 0; j < neurons[n + 1]; ++j) {
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
  const vector<int> neurons;
  const int layers;
public:
  Y(const vector<int> &neurons) : neurons(neurons), layers(neurons.size()), data() {
    for (int l = 0; l < layers; ++l) {
      data.push_back(vector<double>(neurons[l], 0.0));
    }
  }
  vector<double> &operator[](const size_t layer) {
    return data[layer];
  }
  const vector<double> &operator[](const size_t layer) const {
    return data[layer];
  }
  double &at(const size_t layer, const size_t neuron) {
    return data[layer][neuron];
  }
  const double &at(const size_t layer, const size_t neuron) const {
    return data[layer][neuron];
  }
  const int getLayers() const {
    return layers;
  }
  const vector<int> &getNeurons() const {
    return neurons;
  }
  void print() const {
    cout << "Y:" << endl;
    for (int l = 0; l < layers; ++l) {
      for (int i = 0; i < neurons[l]; ++i) {
        cout << " " << data[l][i];
      }
      cout << endl;
    }
  }
};

class Patterns {
private:
  vector<vector<double>> inputs;
  vector<vector<double>> outputs;
  int sz;
public:
  Patterns() : sz(0), inputs(), outputs() {
  }
  void add(const vector<double> &&input, const vector<double> &&output) {
    inputs.emplace_back(input);
    outputs.emplace_back(output);
    ++sz;
  }
  int size() const {
    return sz;
  }
  const vector<vector<double>> &getInputs() const {
    return inputs;
  }
  const vector<vector<double>> &getOutputs() const {
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

// sigmoid is the activation function here
inline double activation(double z) {
  return sigmoid(z);
}

void FeedForward(const W &w, Y &y, const vector<double> &input) {
  // the first layer of Y is the input
  y[0].assign(input.begin(), input.end());
  const int layers = y.getLayers();
  const vector<int> &neurons = y.getNeurons();
  for (int l = 1; l < layers; ++l) {
    for (int j = 0; j < neurons[l]; ++j) {
      double dot_product = 0.0;
      for (int i = 0; i < neurons[l - 1]; ++i) {
        dot_product += w.at(l - 1, i, j) * y.at(l - 1, i);
      }
      y.at(l, j) = sigmoid(dot_product);
    }
  }
}

vector<double> Evaluate(const W &w, const vector<double> &input) {
  Y y(w.getNeurons());
  FeedForward(w, y, input);
  return y[w.getLayers() - 1];
}

void OutputDelta(const vector<double> &o, vector<double> &delta, const Y &y) {
  delta.clear();
  const int lastLayer = y.getLayers() - 1;
  for (size_t i = 0; i < o.size(); ++i) {
    double o_i = o[i];
    double y_i = y.at(lastLayer, i);
    delta.push_back((o_i - y_i) * sigmoid_prime(y_i));
  }
}

void HiddenDelta(const vector<double> delta_next, vector<double> &delta, const W &w, const Y &y, const int layer) {
  delta.clear();
  const int lastLayer = y.getLayers() - 1;
  const int n = y.getNeurons()[layer];
  const size_t m = delta_next.size();
  for (int j = 0; j < n; ++j) {
    double y_i = y.at(layer, j);
    double dot_product = 0.0;
    for (size_t i = 0; i < m; ++i) {
      dot_product += delta_next[i] * w.at(layer, j, i);
    }
    delta.push_back(dot_product * sigmoid_prime(y_i));
  }
}

void Backprop(W &w, const vector<Y> &ys, const vector<vector<double>> &outputs, const double learning_rate) {
  if (ys.empty()) {
    return;
  }
  const int patterns = ys.size();
  const int layers = ys[0].getLayers();
  const vector<int> &nodes = ys[0].getNeurons();
  for (int p = 0; p < patterns; ++p) {
    const Y &y = ys[p];
    const vector<double> &output = outputs[p];
    vector<double> delta;
    OutputDelta(output, delta, y);
    for (int layer = layers - 2; layer >= 0; --layer) {
      for (int k = 0; k < nodes[layer]; ++k) {
        for (int j = 0; j < nodes[layer + 1]; ++j) {
          const double dw = learning_rate * delta[j] * y.at(layer, k);
          w.at(layer, k, j) = dw + w.at(layer, k, j);
        }
      }
      if (layer != 0) {
        HiddenDelta(delta, delta, w, y, layer);
      }
    }
  }
}

string Serialize(const vector<double> &vec) {
  ostringstream ss;
  for (double v : vec) {
    ss << ' ' << v;
  }
  return ss.str();
}

int main(int argc, char** argv) {
  const double learning_rate = 1.0;
  const int epoch_max = 100000;
  const vector<int> neurons{ 2, 3, 4, 1 };
  // XOR
  Patterns patterns;
  patterns.add(vector<double>{1, 1}, vector<double>{0});
  patterns.add(vector<double>{1, 0}, vector<double>{1});
  patterns.add(vector<double>{0, 1}, vector<double>{1});
  patterns.add(vector<double>{0, 0}, vector<double>{0});
  W w(neurons);
  w.print();
  vector<Y> ys(patterns.size(), Y(neurons));
  for (int epoch = 0; epoch < epoch_max; ++epoch) {
    for (int p = 0; p < patterns.size(); ++p) {
      FeedForward(w, ys[p], patterns.getInputs()[p]);
    }
    Backprop(w, ys, patterns.getOutputs(), learning_rate);
    // print intermediate result
    if (epoch % (epoch_max / 10) == 0) {
      cout << endl;
      cout << "====== Epoch " << epoch << " ======" << endl;
      w.print();
      cout << "Validation:" << endl;
      for (const vector<double> &x : patterns.getInputs()) {
        cout << Serialize(x) << ": " << Evaluate(w, x)[0] << endl;
      }
    }
  }
  return 0;
}
