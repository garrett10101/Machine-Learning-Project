#ifndef MLMODEL_H
#define MLMODEL_H

#include <vector>
#include <string>
#include <map>

template <typename T>
class MLModel {
public:
    MLModel() {}
    virtual ~MLModel() {}

    // Load dataset
    void loadDataset(const std::vector<std::vector<T>>& features,
                     const std::vector<T>& target,
                     const std::vector<std::string>& feature_names);

    // Split dataset
    void trainTestSplit(float test_size = 0.2, unsigned int seed = 42);

    // Model-specific methods (must be implemented by subclasses)
    virtual void fit(int epochs = 100, double learning_rate = 0.01) = 0;
    virtual std::vector<T> predict(const std::vector<std::vector<T>>& data) = 0;

    // Common evaluation metrics
    double meanSquaredError(const std::vector<T>& actual, const std::vector<T>& predicted);
    double meanAbsoluteError(const std::vector<T>& actual, const std::vector<T>& predicted);
    double r2Score(const std::vector<T>& actual, const std::vector<T>& predicted);

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

#include "MLModel.tpp"  // Template implementation
#endif
