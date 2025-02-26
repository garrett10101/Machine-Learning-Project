#ifndef MLMODEL_REGRESSOR_H
#define MLMODEL_REGRESSOR_H

#include "MLModel.h"

#include <vector>

class MLModel_Regressor : public MLModel {
public:
    // Constructor
    MLModel_Regressor();

    // Destructor
    virtual ~MLModel_Regressor();

    // Train the regression model with given data
    virtual void train(const std::vector<std::vector<double>>& features, const std::vector<double>& targets) = 0;

    // Predict the target value for given features
    virtual double predict(const std::vector<double>& features) const = 0;

    // Evaluate the regression model with test data
    virtual double evaluate(const std::vector<std::vector<double>>& testFeatures, const std::vector<double>& testTargets) const = 0;

    // Calculate Mean Squared Error (MSE)
    virtual double calculateMSE(const std::vector<double>& predicted, const std::vector<double>& actual) const;

    // Calculate Mean Absolute Error (MAE)
    virtual double calculateMAE(const std::vector<double>& predicted, const std::vector<double>& actual) const;

    // Calculate R-squared (R2) score
    virtual double calculateR2(const std::vector<double>& predicted, const std::vector<double>& actual) const;

    // Calculate Root Mean Squared Error (RMSE)
    virtual double calculateRMSE(const std::vector<double>& predicted, const std::vector<double>& actual) const;
};

#endif // MLMODEL_REGRESSOR_H