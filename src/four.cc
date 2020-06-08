#include <iostream>
#include <stdio.h>
#include <queue>
#include <numeric>
#include <random>
#include <math.h>
#include <Eigen/Dense>
#include "matplotlibcpp.h"

using namespace Eigen;

namespace plt = matplotlibcpp;

std::pair<double, double> simulateSynchronousSystem(double lambda, int message_count, int buffer_len) {
    std::mt19937 mt;
    mt.seed(std::random_device()());
    std::exponential_distribution<> dist(lambda);
    
    int sentMessages = 0;
    std::deque<int> buffer;

    int windowNumber = 0;
    int msgNumber = 0;
    double delay = 0;
    double averageMessage = 0;
    double time = 0;
    int tmp = 0;
  
    while (sentMessages != message_count) {
        time = dist(mt);
        while (time < 1) {
            if (buffer.size() < buffer_len) {
                buffer.push_front(windowNumber);
            }
            time += dist(mt);
        }
        
        if (!buffer.empty() && buffer.back() != windowNumber) {
            int timeIn = buffer.back();
            int timeOut = windowNumber;
            delay += timeOut - timeIn;
            buffer.pop_back();
            sentMessages++;
        }
        windowNumber++;
        averageMessage += buffer.size();
    }
    
    averageMessage /= double(windowNumber);
    delay /= double(message_count);
    return {delay, averageMessage};
}

int factorial(int n) {
  return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}

double get_pr(double lambda, int i) {
    return pow(lambda, i) / double(factorial(i)) * exp(-lambda);
}

double get_row_sum(MatrixXd m, int row) {
    double sum = 0.0;
    for (int i = 0; i < m.cols(); i++) {
        sum += m(row, i);
    }
    return sum;
}

MatrixXd get_transition_matrix(double lambda, int buffer_len) {
    int matrix_size = buffer_len + 1;
    MatrixXd matrix(matrix_size, matrix_size);
    matrix.setZero(matrix_size, matrix_size);
    for (int i = 0; i < matrix_size - 1; i++) {
        matrix(0, i) = matrix(1, i) =  get_pr(lambda, i);
    }
    matrix(0, matrix_size - 1) = 1 - get_row_sum(matrix, 0);
    matrix(1, matrix_size - 2) = 0;
    for (int i = 2; i < matrix_size - 1; i++) {
        int idx = 0;
        for (int j = i - 1; j < matrix_size - 2; j++) {
            matrix(i, j) = get_pr(lambda, idx);
            idx++;
        }
    }
    
    for (int i = 1; i < matrix_size; i++) {
        matrix(i, matrix_size - 2) = 1.0 - get_row_sum(matrix, i);
    }
    MatrixXd matrix_tr = matrix.transpose();
    
    for (int i = 0; i < matrix_size; i++) {
        matrix_tr(i, i) -= 1;
    }
    for (int i = 0; i < matrix_size; i++) {
        matrix_tr(matrix_size - 1, i) = 1;
    }
    return matrix_tr;
} 

void plot_system_stats(int message_count, int buffer_len) {
    std::vector<double> x;
    std::vector<double> d;
    std::vector<double> n;
    std::vector<double> d_theor;
    std::vector<double> n_theor;
    double lambdas[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    
    MatrixXd transition_matrix;
    for (double lambda : lambdas) {
        printf("λ = %f, buf_len = %d\n", lambda, buffer_len);
        std::pair<double, double> res = simulateSynchronousSystem(lambda, message_count, buffer_len);
        transition_matrix = get_transition_matrix(lambda, buffer_len);
        VectorXd vec(buffer_len + 1);
        vec.setZero(buffer_len + 1);
        vec[buffer_len] = 1.0;
        VectorXd pi = transition_matrix.inverse() * vec;
        double avg_n = 0.0;
        for (int i = 0; i < pi.size(); i++) {
            avg_n += i * pi[i];
        }
        double lambda_out = 1 - pi[0];
        x.push_back(lambda);
        d.push_back(res.first);
        d_theor.push_back(avg_n / lambda_out);
        n.push_back(res.second);
        n_theor.push_back(avg_n);
    }
    plt::plot(x, d, {{"label", "d"}});
    plt::plot(x, d_theor, {{"label", "d_theor"}});
    plt::plot(x, n, {{"label", "n"}});
    plt::plot(x, n_theor, {{"label", "n_theor"}});
    plt::legend();
    plt::save("four.png");
}

int main() {
    int message_count = 100000;
    int buffer_len = 10;
    plot_system_stats(message_count, buffer_len);
    return 0;
}