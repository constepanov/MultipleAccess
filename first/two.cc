#include <iostream>
#include <random>
#include <Eigen/Dense>
#include "matplotlibcpp.h"

using namespace Eigen;

namespace plt = matplotlibcpp;

Vector3d simulate(int time, Matrix3d &transitionMatrix) {
  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_real_distribution<float> dist(0.f, 1.f);

  int state0_count = 0;
  int state1_count = 0;
  int state2_count = 0;
  int currentState = 0;
  
  for (int j = 0; j < time; j++) {
    double r = dist(rng);
    if (currentState == 0) {
      if (r < transitionMatrix(currentState, 0)) {
        currentState = 0;
      } else if (r < (transitionMatrix(currentState, 0) + transitionMatrix(currentState, 1))) {
        currentState = 1;
      } else {
        currentState = 2;
      }
    } else if (currentState == 1) {
      if (r < transitionMatrix(currentState, 0)) {
        currentState = 0;
      } else if (r < (transitionMatrix(currentState, 0) + transitionMatrix(currentState, 1))) {
        currentState = 1;
      } else {
        currentState = 2;
      }
    } else {
      if (r < transitionMatrix(currentState, 0)) {
        currentState = 0;
      } else if (r < (transitionMatrix(currentState, 0) + transitionMatrix(currentState, 1))) {
        currentState = 1;
      } else {
        currentState = 2;
      }
    }
    if (currentState == 0) state0_count++;
    if (currentState == 1) state1_count++;
    if (currentState == 2) state2_count++;
  }
  Vector3d distribution(double(state0_count) / time, 
                        double(state1_count) / time, 
                        double(state2_count) / time);
  return distribution;
}

Vector3d stationary_distribution(Matrix3d &transitionMatrix) {
  Vector3d b(0.0, 0.0, 1.0);
  Matrix3d pt = transitionMatrix.transpose();
  pt.row(2).setOnes();
  pt(0, 0) -= 1;
  pt(1, 1) -= 1;
  return pt.colPivHouseholderQr().solve(b);
}

int main() {
  int time = 1000000;
  Matrix3d transitionMatrix;
  transitionMatrix << 0.3, 0.2, 0.5,
                      0.2, 0.4, 0.4,
                      0.1, 0.7, 0.2;
  std::cout << "Simulate distribution:\n" 
            << simulate(time, transitionMatrix) << std::endl;
  std::cout << "Theoretic distribution:\n" 
            << stationary_distribution(transitionMatrix) << std::endl;
}