#include <iostream>
#include <random>
#include <Eigen/Dense>
#include "matplotlibcpp.h"

using namespace Eigen;

namespace plt = matplotlibcpp;

std::pair<double, double> simulate(int time, int iterations, Matrix2d &transitionMatrix) {
  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_real_distribution<float> dist(0.f, 1.f);

  int state0_count = 0;
  int state1_count = 0;
  int currentState = 0;

  for (int i = 0; i < iterations; i++) {
    currentState = 0;
    for (int j = 0; j < time; j++) {
      if (currentState == 0) {
        if (dist(rng) > transitionMatrix(0, 0)) currentState = 1;
      } else {
        if (dist(rng) > transitionMatrix(1, 1)) currentState = 0;
      }
    }
    if (currentState == 0) state0_count++;
    if (currentState == 1) state1_count++;
  }
  return {double(state0_count) / iterations, double(state1_count) / iterations};
}

std::vector<std::pair<double, double>> theoretic_probabilities(int time, Matrix2d &transitionMatrix) {
  std::vector<std::pair<double, double>> th_prob;
  th_prob.push_back({1.0, 0.0});
  RowVector2d p0(1.0, 0.0);
  for (int t = 1; t < time; t++) {
    p0 = p0 * transitionMatrix;
    th_prob.push_back({p0(0), p0(1)});
  }
  return th_prob;
}

int main() {
  int time = 100;
  int iterations = 100000;
  Matrix2d transitionMatrix;
  transitionMatrix << 0.8, 0.2,
                      0.6, 0.4;
  
  std::vector<double> pr0(time);
  std::vector<double> pr1(time);
  std::vector<double> x(time);

  std::vector<std::pair<double, double>> th_pr = theoretic_probabilities(time, transitionMatrix);
  std::vector<double> th_pr0(time);
  std::vector<double> th_pr1(time);
  
  for (int t = 0; t < time; t++) {
    std::pair<double, double> prob = simulate(t, iterations, transitionMatrix);
    pr0[t] = prob.first;
    pr1[t] = prob.second;
    th_pr0[t] = th_pr[t].first;
    th_pr1[t] = th_pr[t].second;
    x[t] = t;
  }

  plt::plot(x, pr0, {{"label", "pr0"}});
  plt::plot(x, pr1, {{"label", "pr1"}});
  plt::plot(x, th_pr0, {{"label", "th_pr0"}});
  plt::plot(x, th_pr1, {{"label", "th_pr1"}});
  plt::legend();
  plt::save("one.png");
}
