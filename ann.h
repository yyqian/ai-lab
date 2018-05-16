//
// Created by Yinyin Qian on 5/16/18.
//

#ifndef AI_LAB_ANN_H
#define AI_LAB_ANN_H
#include <vector>
#include <cmath>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <random>

class W {
 private:
  std::vector<std::vector<double>> data;
  const int layers;
  const std::vector<int> neurons;
 public:
  W(const std::vector<int> &neurons) : layers(static_cast<const int>(neurons.size())), neurons(neurons), data() {
    for (int i = 0; i < layers - 1; ++i) {
      data.emplace_back(static_cast<unsigned long>(neurons[i] * neurons[i + 1]), 0.0);
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
  const std::vector<int> &getNeurons() const {
    return neurons;
  }
  void randomize() {
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    for (std::vector<double> &v : data) {
      for (double &ele : v) {
        ele = distribution(generator);
      }
    }
  }
  void print() const {
    for (int n = 0; n < layers - 1; ++n) {
      std::cout << "W[" << n << "]:" << std::endl;
      for (int i = 0; i < neurons[n]; ++i) {
        for (int j = 0; j < neurons[n + 1]; ++j) {
          std::cout << " " << at(n, i, j);
        }
        std::cout << std::endl;
      }
    }
  }
};

class Y {
 private:
  std::vector<std::vector<double>> data;
  const std::vector<int> neurons;
  const int layers;
 public:
  Y(const std::vector<int> &neurons) : neurons(neurons), layers(static_cast<const int>(neurons.size())), data() {
    for (int l = 0; l < layers; ++l) {
      data.emplace_back(neurons[l], 0.0);
    }
  }
  std::vector<double> &operator[](const size_t layer) {
    return data[layer];
  }
  const std::vector<double> &operator[](const size_t layer) const {
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
  const std::vector<int> &getNeurons() const {
    return neurons;
  }
  void print() const {
    std::cout << "Y:" << std::endl;
    for (int l = 0; l < layers; ++l) {
      for (int i = 0; i < neurons[l]; ++i) {
        std::cout << " " << data[l][i];
      }
      std::cout << std::endl;
    }
  }
};

class Patterns {
 private:
  std::vector<std::vector<double>> inputs;
  std::vector<std::vector<double>> outputs;
  int sz;
 public:
  Patterns() : sz(0), inputs(), outputs() {
  }
  void add(const std::vector<double> &&input, const std::vector<double> &&output) {
    inputs.emplace_back(input);
    outputs.emplace_back(output);
    ++sz;
  }
  int size() const {
    return sz;
  }
  const std::vector<std::vector<double>> &getInputs() const {
    return inputs;
  }
  const std::vector<std::vector<double>> &getOutputs() const {
    return outputs;
  }
};

#endif //AI_LAB_ANN_H
