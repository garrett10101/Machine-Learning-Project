#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../ML_Framework/Regressors/LinReg/LinearReg.h"

// Test cases
TEST_CASE("LinearReg Fit") {
    LinearReg model;
    std::vector<double> X = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {2.0, 4.0, 6.0, 8.0, 10.0};
    model.fit(X, y);
    CHECK(true); // Check if fit runs without errors
}

TEST_CASE("LinearReg Predict") {
    LinearReg model;
    std::vector<double> X = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {2.0, 4.0, 6.0, 8.0, 10.0};
    model.fit(X, y);
    std::vector<double> predictions = model.predict(X);
    CHECK(predictions == y); // Check if predictions match the expected values
}

TEST_CASE("LinearReg Score") {
    LinearReg model;
    std::vector<double> X = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {2.0, 4.0, 6.0, 8.0, 10.0};
    model.fit(X, y);
    double score = model.score(X, y);
    CHECK(score == doctest::Approx(1.0)); // Check if the score is close to 1.0
}

TEST_CASE("LinearReg Tune") {
    LinearReg model;
    model.tune(0.01, 1000);
    CHECK(true); // Check if tune runs without errors
}