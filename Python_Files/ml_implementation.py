# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tpot import TPOTRegressor
import joblib

# ML Implementation code
def train_model(data, target):
    X = data.drop(target, axis=1)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Model Mean Squared Error: {mse}")
    
    model.export('best_model_pipeline.py')
    joblib.dump(model.fitted_pipeline_, 'model.pkl')
    return model
# ML Implementation code that shows metrics and saves it to txt file
def get_metrics(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    with open('metrics.txt', 'w') as f:
        f.write(f"Model Mean Squared Error: {mse}")
    print(f"Model Mean Squared Error: {mse}")
if __name__ == "__main__":
    from data_preprocessing import load_data, preprocess_data
    
    file_path = 'data/Final_Data_Frame.csv'
    data = load_data(file_path)
    processed_data = preprocess_data(data)
    
    tpot_model = train_model(processed_data, 'TDS')
    get_metrics(tpot_model, processed_data.drop('TDS', axis=1), processed_data['TDS'])