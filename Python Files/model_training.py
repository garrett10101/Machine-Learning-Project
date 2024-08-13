#File that takes in data and trains various models and exports model to models folder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pytz
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#function that gets every regression model from sklearn and saves to a dictionary
def get_models():
    models = {}
    models['Linear Regression'] = LinearRegression()
    models['Random Forest'] = RandomForestRegressor()
    return models
#function that trains models and returns a dictionary of models and their scores
def train_models(X_train, y_train):
    models = get_models()
    model_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        model_scores[name] = model.score(X_train, y_train)
    return model_scores
#function that prints out scores and other metrics for each model
def print_scores(model_scores):
    for name, score in model_scores.items():
        print(f'{name} Score: {score}')