#include "LinearReg.h"
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>

LinearReg::LinearReg() : intercept(0.0), slope(0.0) {}

void LinearReg::fit(const std::vector<double>& X, const std::vector<double>& y) {
    if (X.size() != y.size()) {
        throw std::invalid_argument("The size of X and y must be the same.");
    }

    double n = X.size();
    double sumX = std::accumulate(X.begin(), X.end(), 0.0);
    double sumY = std::accumulate(y.begin(), y.end(), 0.0);
    double sumXY = 0.0;
    double sumX2 = 0.0;

    for (size_t i = 0; i < n; ++i) {
        sumXY += X[i] * y[i];
        sumX2 += X[i] * X[i];
    }

    slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    intercept = (sumY - slope * sumX) / n;
}

std::vector<double> LinearReg::predict(const std::vector<double>& X) const {
    std::vector<double> predictions;
    predictions.reserve(X.size());

    for (const auto& x : X) {
        predictions.push_back(intercept + slope * x);
    }

    return predictions;
}

double LinearReg::score(const std::vector<double>& X, const std::vector<double>& y) const {
    if (X.size() != y.size()) {
        throw std::invalid_argument("The size of X and y must be the same.");
    }

    std::vector<double> predictions = predict(X);
    double meanY = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
    double ssTotal = 0.0;
    double ssResidual = 0.0;

    for (size_t i = 0; i < y.size(); ++i) {
        ssTotal += (y[i] - meanY) * (y[i] - meanY);
        ssResidual += (y[i] - predictions[i]) * (y[i] - predictions[i]);
    }

    return 1 - (ssResidual / ssTotal);
}

void LinearReg::tune(double learning_rate, int iterations) {
    // Placeholder for tuning implementation
    // This function can be used to implement gradient descent or other optimization techniques
    std::cout << "Tuning with learning rate: " << learning_rate << " for " << iterations << " iterations." << std::endl;
}