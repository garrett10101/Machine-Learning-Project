cmake_minimum_required(VERSION 3.10)
project(UnitTests)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the executable
add_executable(ml_model_test ml_model_test.cpp)
add_executable(ml_model_regressor_test ml_model_regressor_test.cpp)
add_executable(linear_reg_test linear_reg_test.cpp)