#include "MLModel.h"
#include <algorithm>
#include <random>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <iostream>

template <typename T>
void MLModel<T>::loadDataset(const std::vector<std::vector<T>>& features,
                             const std::vector<T>& target,
                             const std::vector<std::string>& feature_names) {
    if (features.size() != target.size())
        throw std::invalid_argument("Feature and target size mismatch.");
    this->features = features;
    this->target = target;
    this->feature_names = feature_names;
}

template <typename T>
void MLModel<T>::trainTestSplit(float test_size, unsigned int seed) {
    size_t total = features.size(), test_n = total * test_size;
    std::vector<size_t> indices(total);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine(seed));

    for (size_t i = 0; i < total; ++i) {
        if (i < test_n) {
            X_test.push_back(features[indices[i]]);
            y_test.push_back(target[indices[i]]);
        } else {
            X_train.push_back(features[indices[i]]);
            y_train.push_back(target[indices[i]]);
        }
    }
}

template <typename T>
double MLModel<T>::meanSquaredError(const std::vector<T>& actual, const std::vector<T>& predicted) {
    double mse = 0.0;
    for (size_t i = 0; i < actual.size(); ++i)
        mse += pow(actual[i] - predicted[i], 2);
    return mse / actual.size();
}

template <typename T>
double MLModel<T>::meanAbsoluteError(const std::vector<T>& actual, const std::vector<T>& predicted) {
    double mae = 0.0;
    for (size_t i = 0; i < actual.size(); ++i)
        mae += abs(actual[i] - predicted[i]);
    return mae / actual.size();
}

template <typename T>
double MLModel<T>::r2Score(const std::vector<T>& actual, const std::vector<T>& predicted) {
    double ss_res = 0, ss_tot = 0, mean = accumulate(actual.begin(), actual.end(), 0.0) / actual.size();
    for (size_t i = 0; i < actual.size(); ++i) {
        ss_res += pow(actual[i] - predicted[i], 2);
        ss_tot += pow(actual[i] - mean, 2);
    }
    return 1 - (ss_res / ss_tot);
}

template <typename T>
std::vector<std::vector<T>> MLModel<T>::getTrainData() const { return X_train; }

template <typename T>
std::vector<std::vector<T>> MLModel<T>::getTestData() const { return X_test; }

template <typename T>
std::vector<T> MLModel<T>::getTrainTarget() const { return y_train; }

template <typename T>
std::vector<T> MLModel<T>::getTestTarget() const { return y_test; }

template <typename T>
std::vector<std::string> MLModel<T>::getFeatureNames() const { return feature_names; }
