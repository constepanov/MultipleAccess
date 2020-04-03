#include <iostream>
#include <random>
#include <Eigen/Dense>
#include "matplotlibcpp.h"

using namespace Eigen;

namespace plt = matplotlibcpp;

int change_state(double rnd, int currentState, Matrix3d &transitionMatrix) {
  int new_state = 0;
  if (rnd < transitionMatrix(currentState, 0)) {
    new_state = 0;
  } else if (rnd < (transitionMatrix(currentState, 0) + transitionMatrix(currentState, 1))) {
    new_state = 1;
  } else {
    new_state = 2;
  }
  return new_state;
}

double simulate(int iterations, int state, Matrix3d &transitionMatrix) {
  std::mt19937 mt;
  mt.seed(std::random_device()());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
 
  int count = 0;
  for (int i = 0; i < iterations; i++) {
    int currentState = state;
    int j = 0;
    for (;; j++) {
      double rnd = dist(mt);
      currentState = change_state(rnd, currentState, transitionMatrix);
      if (currentState == 2) break;
    }
    count += j;
  }
  return double(count) / iterations;
}

double theoretic_average_time(int state, Matrix3d &transitionMatrix) {
  Vector3d b(1, 1, 1);
  Matrix3d a = -transitionMatrix;
  a(0, 0) += 1;
  a(1, 1) += 1;
  return (a.inverse() * b)(state);
}

int main() {
  int iterations = 1000000;
  Matrix3d transitionMatrix;
  transitionMatrix << 0.7, 0.295, 0.005,
                      0.5, 0.495, 0.005,
                      0, 0, 1;
  int state = 0;
  std::cout << "Simulate average time: "  
            << simulate(iterations, state, transitionMatrix) << std::endl;
  std::cout << "Theoretic average time: " 
            << theoretic_average_time(state, transitionMatrix) << std::endl;
  return 0;
}