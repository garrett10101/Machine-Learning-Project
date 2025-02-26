#ifndef MLMODEL_H
#define MLMODEL_H

#include <vector>
#include <string>

template <typename T>
class MLModel {
public:
    // Constructor
    MLModel();

    // Load dataset from a file
    virtual bool loadDataset(const std::string& filename) = 0;

    // Train the model with the dataset
    virtual void train() = 0;

    // Fit the model with the given data and labels
    virtual void fit(const std::vector<T>& data, const std::vector<T>& labels) = 0;

    // Evaluate the model with the given test data and labels
    virtual double evaluate(const std::vector<T>& testData, const std::vector<T>& testLabels) const = 0;

    // Tune the model's hyperparameters
    virtual void tune(const std::vector<T>& validationData, const std::vector<T>& validationLabels) = 0;

    // Predict the output for the given input data
    virtual std::vector<T> predict(const std::vector<T>& inputData) const = 0;

    // Virtual destructor
    virtual ~MLModel() {}
};

#include "MLModel.tpp"
#endif // MLMODEL_H