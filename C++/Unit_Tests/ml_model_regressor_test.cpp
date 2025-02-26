#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../ML_Framework/Regressors/MLModel_Regressor.h"

// Define a mock class for testing
template <typename T>
class MockMLModel_Regressor : public MLModel_Regressor<T> {
public:
    void train(const std::vector<std::vector<T>>& features, const std::vector<T>& targets) override {
        // Mock implementation
    }

    std::vector<T> predict(const std::vector<std::vector<T>>& features) const override {
        // Mock implementation
        return features[0];
    }

    T mean_squared_error(const std::vector<T>& true_values, const std::vector<T>& predicted_values) const override {
        // Mock implementation
        return 0.0;
    }

    T mean_absolute_error(const std::vector<T>& true_values, const std::vector<T>& predicted_values) const override {
        // Mock implementation
        return 0.0;
    }

    T r2_score(const std::vector<T>& true_values, const std::vector<T>& predicted_values) const override {
        // Mock implementation
        return 1.0;
    }
};

// Test cases
TEST_CASE("MLModel_Regressor Train") {
    MockMLModel_Regressor<int> model;
    std::vector<std::vector<int>> features = {{1, 2}, {3, 4}, {5, 6}};
    std::vector<int> targets = {1, 2, 3};
    model.train(features, targets);
    CHECK(true);
}

TEST_CASE("MLModel_Regressor Predict") {
    MockMLModel_Regressor<int> model;
    std::vector<std::vector<int>> features = {{1, 2}, {3, 4}, {5, 6}};
    std::vector<int> predictions = model.predict(features);
    CHECK(predictions == features[0]);
}

TEST_CASE("MLModel_Regressor MeanSquaredError") {
    MockMLModel_Regressor<int> model;
    std::vector<int> true_values = {1, 2, 3};
    std::vector<int> predicted_values = {1, 2, 3};
    CHECK(model.mean_squared_error(true_values, predicted_values) == 0.0);
}

TEST_CASE("MLModel_Regressor MeanAbsoluteError") {
    MockMLModel_Regressor<int> model;
    std::vector<int> true_values = {1, 2, 3};
    std::vector<int> predicted_values = {1, 2, 3};
    CHECK(model.mean_absolute_error(true_values, predicted_values) == 0.0);
}

TEST_CASE("MLModel_Regressor R2Score") {
    MockMLModel_Regressor<int> model;
    std::vector<int> true_values = {1, 2, 3};
    std::vector<int> predicted_values = {1, 2, 3};
    CHECK(model.r2_score(true_values, predicted_values) == 1.0);
}