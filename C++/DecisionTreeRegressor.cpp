#include "DecisionTreeRegressor.h"
#include <iostream>
#include <limits>
#include <cmath>
#include <numeric>

// Constructor
template <typename T>
DecisionTreeRegressor<T>::DecisionTreeRegressor() : treeRoot(nullptr) {}

// Destructor
template <typename T>
DecisionTreeRegressor<T>::~DecisionTreeRegressor() {
    delete treeRoot;
}

// Recursive tree building
template <typename T>
void buildTree(Node<T>*& node, const std::vector<std::vector<T>>& X, const std::vector<T>& y,
               int depth, int max_depth, int min_samples_split) {
    if (depth >= max_depth || X.size() <= min_samples_split) {
        T sum = std::accumulate(y.begin(), y.end(), 0.0);
        node->prediction = sum / y.size();
        node->isLeaf = true;
        return;
    }

    double best_mse = std::numeric_limits<double>::max();
    size_t best_feature = 0;
    T best_value = 0;

    for (size_t feature = 0; feature < X[0].size(); ++feature) {
        for (const auto& row : X) {
            T split = row[feature];
            std::vector<T> left_y, right_y;
            for (size_t i = 0; i < X.size(); ++i) {
                if (X[i][feature] <= split)
                    left_y.push_back(y[i]);
                else
                    right_y.push_back(y[i]);
            }

            if (left_y.empty() || right_y.empty()) continue;

            T left_mean = std::accumulate(left_y.begin(), left_y.end(), 0.0) / left_y.size();
            T right_mean = std::accumulate(right_y.begin(), right_y.end(), 0.0) / right_y.size();

            double mse = 0;
            for (auto val : left_y) mse += pow(val - left_mean, 2);
            for (auto val : right_y) mse += pow(val - right_mean, 2);
            mse /= y.size();

            if (mse < best_mse) {
                best_mse = mse;
                best_feature = feature;
                best_value = split;
            }
        }
    }

    if (best_mse == std::numeric_limits<double>::max()) {
        node->prediction = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
        node->isLeaf = true;
        return;
    }

    node->isLeaf = false;
    node->featureIndex = best_feature;
    node->splitValue = best_value;

    std::vector<std::vector<T>> left_X, right_X;
    std::vector<T> left_y, right_y;
    for (size_t i = 0; i < X.size(); ++i) {
        if (X[i][best_feature] <= best_value) {
            left_X.push_back(X[i]);
            left_y.push_back(y[i]);
        } else {
            right_X.push_back(X[i]);
            right_y.push_back(y[i]);
        }
    }

    node->left = new Node<T>();
    node->right = new Node<T>();
    buildTree(node->left, left_X, left_y, depth + 1, max_depth, min_samples_split);
    buildTree(node->right, right_X, right_y, depth + 1, max_depth, min_samples_split);
}

// Fit method
template <typename T>
void DecisionTreeRegressor<T>::fit(int max_depth, int min_samples_split) {
    if (treeRoot) delete treeRoot;

    treeRoot = new Node<T>();
    buildTree(treeRoot, this->X_train, this->y_train, 0, max_depth, min_samples_split);
}

// Predict method
template <typename T>
std::vector<T> DecisionTreeRegressor<T>::predict(const std::vector<std::vector<T>>& data) {
    std::vector<T> predictions;
    for (const auto& row : data)
        predictions.push_back(predictSingle(treeRoot, row));
    return predictions;
}

// Explicit instantiation
template class DecisionTreeRegressor<int>;
template class DecisionTreeRegressor<float>;
template class DecisionTreeRegressor<double>;
