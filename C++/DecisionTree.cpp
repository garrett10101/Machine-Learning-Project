#include "DecisionTree.h"
#include <algorithm>
#include <random>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <iostream>

template <typename T>
DecisionTree<T>::DecisionTree() {}

template <typename T>
DecisionTree<T>::~DecisionTree() {}

template <typename T>
void DecisionTree<T>::loadDataset(const std::vector<std::vector<T>>& features,
                                  const std::vector<T>& target,
                                  const std::vector<std::string>& feature_names) {
    if(features.size() != target.size())
        throw std::invalid_argument("Feature and target size mismatch.");
    this->features = features;
    this->target = target;
    this->feature_names = feature_names;
}

template <typename T>
void DecisionTree<T>::trainTestSplit(float test_size, unsigned int seed) {
    size_t total = features.size(), test_n = total * test_size;
    std::vector<size_t> indices(total);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine(seed));

    for(size_t i=0; i<total; ++i){
        if(i < test_n){
            X_test.push_back(features[indices[i]]);
            y_test.push_back(target[indices[i]]);
        } else {
            X_train.push_back(features[indices[i]]);
            y_train.push_back(target[indices[i]]);
        }
    }
}

template <typename T>
float DecisionTree<T>::accuracy(const std::vector<T>& actual, const std::vector<T>& pred) {
    size_t correct = 0, n = actual.size();
    for(size_t i = 0; i < n; ++i)
        if(actual[i] == pred[i]) correct++;
    return static_cast<float>(correct) / n;
}

template <typename T>
double DecisionTree<T>::meanSquaredError(const std::vector<T>& actual, const std::vector<T>& pred) {
    double mse = 0.0;
    for(size_t i=0; i<actual.size(); ++i)
        mse += pow(actual[i] - pred[i], 2);
    return mse / actual.size();
}

template <typename T>
double DecisionTree<T>::meanAbsoluteError(const std::vector<T>& actual, const std::vector<T>& pred) {
    double mae = 0.0;
    for(size_t i=0; i<actual.size(); ++i)
        mae += abs(actual[i] - pred[i]);
    return mae / actual.size();
}

template <typename T>
double DecisionTree<T>::r2Score(const std::vector<T>& actual, const std::vector<T>& pred) {
    double ss_res=0, ss_tot=0, mean=accumulate(actual.begin(),actual.end(),0.0)/actual.size();
    for(size_t i=0;i<actual.size();++i){
        ss_res+=pow(actual[i]-pred[i],2);
        ss_tot+=pow(actual[i]-mean,2);
    }
    return 1-(ss_res/ss_tot);
}

template <typename T>
void DecisionTree<T>::hyperparameterTuning(int max_iter, float target_accuracy, int max_depth_range, int min_samples_range){
    float best_acc = 0;
    int best_depth = 1, best_samples = 2;

    for(int i = 1; i <= max_iter && best_acc < target_accuracy; ++i){
        int depth = rand() % max_depth_range + 1;
        int samples = rand() % (min_samples_range-2) + 2;

        fit(depth, samples);
        auto preds = predict(X_test);
        float acc = accuracy(y_test, preds);

        std::cout << "Iteration " << i << " | Depth: " << depth << ", Min Samples: " << samples 
                  << ", Accuracy: " << acc << std::endl;

        if(acc > best_acc){
            best_acc = acc;
            best_depth = depth;
            best_samples = samples;
        }
        if(best_acc >= target_accuracy)
            break;
    }

    std::cout << "Best params -> Depth: " << best_depth 
              << ", Min Samples: " << best_samples 
              << ", Accuracy: " << best_acc << std::endl;

    fit(best_depth, best_samples);
}

template <typename T>
std::vector<std::vector<T>> DecisionTree<T>::getTrainData() const {
    return X_train;
}

template <typename T>
std::vector<std::vector<T>> DecisionTree<T>::getTestData() const {
    return X_test;
}

template <typename T>
std::vector<T> DecisionTree<T>::getTrainTarget() const {
    return y_train;
}

template <typename T>
std::vector<T> DecisionTree<T>::getTestTarget() const {
    return y_test;
}

template <typename T>
std::vector<std::string> DecisionTree<T>::getFeatureNames() const {
    return feature_names;
}

// Explicit instantiations
template class DecisionTree<int>;
template class DecisionTree<float>;
template class DecisionTree<double>;
