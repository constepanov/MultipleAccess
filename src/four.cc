#include <iostream>
#include <queue>
#include <random>
#include <Eigen/Dense>
#include "matplotlibcpp.h"

using namespace Eigen;

namespace plt = matplotlibcpp;

void simulateSynchronousSystem(double lambda, int message_count, int buffer_len) {
    int sentMessages = 0;
    std::queue<int> buffer;
    while (sentMessages < message_count) {
        sentMessages++;
        // TODO
    }
}

int main() {
    double lambda = 0.1;
    int message_count = 10000;
    int buffer_len = 5;
    simulateSynchronousSystem(lambda, message_count, buffer_len);
    return 0;
}