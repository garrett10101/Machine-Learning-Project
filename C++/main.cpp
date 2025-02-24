#include "CSVReader.h"
#include "DecisionTreeRegressor.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <thread>
#include <future>
#include <memory>

// Read and preprocess data
void readData(const std::string& filename, std::vector<std::vector<double>>& features,
              std::vector<double>& target, std::vector<std::string>& feature_names) {
    CSVReader reader(filename);
    auto data = reader.getData();
    auto header = reader.getHeader();

    // Identify target variable (TDS) and feature columns
    int target_col = -1;
    for (size_t i = 0; i < header.size(); ++i) {
        if (header[i] == "TDS") {
            target_col = i;
        } else {
            feature_names.push_back(header[i]);
        }
    }

    if (target_col == -1) {
        std::cerr << "Target column 'TDS' not found." << std::endl;
        return;
    }

    // Populate features and target vector
    for (const auto& row : data) {
        std::vector<double> feature_row;
        for (size_t i = 0; i < row.size(); ++i) {
            double value = std::stod(row[i]);
            if (i == target_col)
                target.push_back(value);
            else
                feature_row.push_back(value);
        }
        features.push_back(feature_row);
    }
}

// Parallelized Training Function
void trainModel(DecisionTreeRegressor<double>& model, int max_depth, int min_samples) {
    model.fit(max_depth, min_samples);
}

int main() {
    std::string filename = "cleaned_output_data.csv";
    std::vector<std::vector<double>> features;
    std::vector<double> target;
    std::vector<std::string> feature_names;

    // Read data
    readData(filename, features, target, feature_names);

    // Use smart pointer for automatic memory management
    std::unique_ptr<DecisionTreeRegressor<double>> dt = std::make_unique<DecisionTreeRegressor<double>>();

    dt->loadDataset(features, target, feature_names);
    dt->trainTestSplit(0.2);

    // Train model using async for multi-threading
    std::future<void> train_future = std::async(std::launch::async, trainModel, std::ref(*dt), 5, 10);
    train_future.wait(); // Ensure training completes before proceeding

    // Parallel Prediction using OpenMP
    std::vector<std::vector<double>> test_data = dt->getTestData();
    std::vector<double> predictions(test_data.size());

    #pragma omp parallel for
    for (size_t i = 0; i < test_data.size(); ++i) {
        predictions[i] = dt->predict({test_data[i]})[0];
    }

    std::cout << "MSE: " << dt->meanSquaredError(dt->getTestTarget(), predictions) << std::endl;
    std::cout << "R2 Score: " << dt->r2Score(dt->getTestTarget(), predictions) << std::endl;

    return 0;
}
