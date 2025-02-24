#include "DecisionTree.h"
#include <iostream>
#include <random>
#include <algorithm>

// Constructor
template <typename T>
DecisionTree<T>::DecisionTree() {}

// Destructor
template <typename T>
DecisionTree<T>::~DecisionTree() {}

// Hyperparameter tuning for decision trees
template <typename T>
void DecisionTree<T>::hyperparameterTuning(int max_iter, float target_accuracy, int max_depth_range, int min_samples_range) {
    float best_acc = 0;
    int best_depth = 1, best_samples = 2;

    for (int i = 1; i <= max_iter && best_acc < target_accuracy; ++i) {
        int depth = rand() % max_depth_range + 1;
        int samples = rand() % (min_samples_range - 2) + 2;

        fit(depth, samples);
        auto preds = this->predict(this->getTestData());
        float acc = this->accuracy(this->getTestTarget(), preds);

        std::cout << "Iteration " << i << " | Depth: " << depth << ", Min Samples: " << samples
                  << ", Accuracy: " << acc << std::endl;

        if (acc > best_acc) {
            best_acc = acc;
            best_depth = depth;
            best_samples = samples;
        }
        if (best_acc >= target_accuracy)
            break;
    }

    std::cout << "Best params -> Depth: " << best_depth
              << ", Min Samples: " << best_samples
              << ", Accuracy: " << best_acc << std::endl;

    fit(best_depth, best_samples);
}

// Explicit template instantiation
template class DecisionTree<int>;
template class DecisionTree<float>;
template class DecisionTree<double>;
