#ifndef MLMODEL_CLASSIFIERS_H
#define MLMODEL_CLASSIFIERS_H

#include "MLModel.h"

class MLModel_Classifiers : public MLModel {
public:
    // Constructor
    MLModel_Classifiers();

    // Destructor
    virtual ~MLModel_Classifiers();

    // Method to train the classifier
    virtual void train(const std::vector<std::vector<double>>& data, const std::vector<int>& labels) = 0;

    // Method to predict the class of a given sample
    virtual int predict(const std::vector<double>& sample) const = 0;

    // Method to evaluate the classifier on a test dataset
    virtual double evaluate(const std::vector<std::vector<double>>& test_data, const std::vector<int>& test_labels) const = 0;
};

#endif // MLMODEL_CLASSIFIERS_H