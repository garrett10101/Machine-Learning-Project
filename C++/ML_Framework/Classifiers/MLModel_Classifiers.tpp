#ifndef MLMODEL_CLASSIFIERS_TPP
#define MLMODEL_CLASSIFIERS_TPP

#include "MLModel_Classifiers.h"
#include <vector>
#include <algorithm>
#include <numeric>

template <typename T>
MLModel_Classifiers<T>::MLModel_Classifiers() {
    // Constructor implementation
}

template <typename T>
MLModel_Classifiers<T>::~MLModel_Classifiers() {
    // Destructor implementation
}

template <typename T>
double MLModel_Classifiers<T>::accuracy(const std::vector<T>& true_labels, const std::vector<T>& predicted_labels) {
    int correct_predictions = 0;
    for (size_t i = 0; i < true_labels.size(); ++i) {
        if (true_labels[i] == predicted_labels[i]) {
            ++correct_predictions;
        }
    }
    return static_cast<double>(correct_predictions) / true_labels.size();
}

template <typename T>
double MLModel_Classifiers<T>::f1_score(const std::vector<T>& true_labels, const std::vector<T>& predicted_labels) {
    double precision = this->precision(true_labels, predicted_labels);
    double recall = this->recall(true_labels, predicted_labels);
    return 2 * (precision * recall) / (precision + recall);
}

template <typename T>
double MLModel_Classifiers<T>::precision(const std::vector<T>& true_labels, const std::vector<T>& predicted_labels) {
    int true_positive = 0;
    int false_positive = 0;
    for (size_t i = 0; i < true_labels.size(); ++i) {
        if (predicted_labels[i] == 1) {
            if (true_labels[i] == 1) {
                ++true_positive;
            } else {
                ++false_positive;
            }
        }
    }
    return static_cast<double>(true_positive) / (true_positive + false_positive);
}

template <typename T>
double MLModel_Classifiers<T>::recall(const std::vector<T>& true_labels, const std::vector<T>& predicted_labels) {
    int true_positive = 0;
    int false_negative = 0;
    for (size_t i = 0; i < true_labels.size(); ++i) {
        if (true_labels[i] == 1) {
            if (predicted_labels[i] == 1) {
                ++true_positive;
            } else {
                ++false_negative;
            }
        }
    }
    return static_cast<double>(true_positive) / (true_positive + false_negative);
}

#endif // MLMODEL_CLASSIFIERS_TPP