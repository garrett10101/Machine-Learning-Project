#ifndef DECISIONTREE_H
#define DECISIONTREE_H

#include "MLModel.h"

template <typename T>
class DecisionTree : public MLModel<T> {
public:
    DecisionTree();
    virtual ~DecisionTree();

    // Override fit and predict methods
    virtual void fit(int max_depth = 5, int min_samples_split = 2) override = 0;
    virtual std::vector<T> predict(const std::vector<std::vector<T>>& data) override = 0;

    // Hyperparameter tuning (specific to trees)
    void hyperparameterTuning(int max_iter, float target_accuracy, int max_depth_range, int min_samples_range);
};

#endif
