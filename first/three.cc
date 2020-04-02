#include <iostream>
#include <random>
#include <Eigen/Dense>
#include "matplotlibcpp.h"

using namespace Eigen;

namespace plt = matplotlibcpp;
/*
  Среднее время достижения - количество шагов / количество итераций
*/

double simulate(int time, int iterations, Matrix3d &transitionMatrix) {
  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
 
  int count = 0;
  for (int i = 0; i < iterations; i++) {
    int currentState = 0;
    int j = 0;
    for (; j < time; j++) {
      double r = dist(rng);
      if (currentState == 0) {
        if (r < transitionMatrix(currentState, 0)) {
          currentState = 0;
        } else if (r < (transitionMatrix(currentState, 0) + transitionMatrix(currentState, 1))) {
          currentState = 1;
        } else {
          currentState = 2;
          break;
        }
      } else if (currentState == 1) {
        if (r < transitionMatrix(currentState, 0)) {
          currentState = 0;
        } else if (r < (transitionMatrix(currentState, 0) + transitionMatrix(currentState, 1))) {
          currentState = 1;
        } else {
          currentState = 2;
          break;
        }
      } else {
        break;
      }
    }
    count += j;
  }
  return double(count) / iterations;
}

int main() {
  int time = 1000000;
  Matrix3d transitionMatrix;
  transitionMatrix << 0.2, 0.75, 0.05,
                      0.6, 0.3, 0.1,
                      0, 0, 1;
  std::cout << simulate(100, 10000, transitionMatrix) << std::endl;
  return 0;
}