#ifndef DECISIONTREE_H
#define DECISIONTREE_H

#include <vector>
#include <string>
#include <map>

template <typename T>
class DecisionTree {
public:
    DecisionTree();
    virtual ~DecisionTree();

    void loadDataset(const std::vector<std::vector<T>>& features,
                     const std::vector<T>& target,
                     const std::vector<std::string>& feature_names);

    void trainTestSplit(float test_size = 0.2, unsigned int seed = 42);

    virtual void fit(int max_depth = 5, int min_samples_split = 2) = 0;
    virtual std::vector<T> predict(const std::vector<std::vector<T>>& data) = 0;

    // Classification metrics
    float accuracy(const std::vector<T>& actual, const std::vector<T>& predicted);

    // Regression metrics
    double meanSquaredError(const std::vector<T>& actual, const std::vector<T>& predicted);
    double meanAbsoluteError(const std::vector<T>& actual, const std::vector<T>& predicted);
    double r2Score(const std::vector<T>& actual, const std::vector<T>& predicted);

    // Hyperparameter tuning
    void hyperparameterTuning(int max_iter, float target_accuracy, int max_depth_range, int min_samples_range);

    std::vector<std::vector<T>> getTrainData() const;
    std::vector<std::vector<T>> getTestData() const;
    std::vector<T> getTrainTarget() const;
    std::vector<T> getTestTarget() const;
    std::vector<std::string> getFeatureNames() const;

protected:
    std::vector<std::vector<T>> features;
    std::vector<T> target;
    std::vector<std::string> feature_names;

    std::vector<std::vector<T>> X_train, X_test;
    std::vector<T> y_train, y_test;
};

#endif
