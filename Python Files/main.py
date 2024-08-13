#file that gets json data from wunderground api, uses config file to match headers from model to json headers, predicts TDS and Water Temp, and appends to a database file (later on a database)
import requests
import json
import pandas as pd
import numpy as np

def get_data():
    url = 'https://api.weather.com/v3/wx/conditions/hourly/7day?geocode=33.44,-94.04&format=json&units=e&language=en-US
    api_key = 'your_api_key'
    headers = {'apikey': api_key}
    response = requests.get(url, headers=headers)
    data = response.json()
    return data
#function that matches json data from json_model.config file to model headers
def match_headers(data, config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    df = pd.DataFrame(data)
    df = df[config['model_headers']]
    return df
#function that predicts TDS and Water Temp and appends to a database file, put a timestamp as the index of the database file
def predict_and_append(df, model_file, database_file):
    model = pd.read_pickle(model_file)
    predictions = model.predict(df)
    df['TDS'] = predictions[:, 0]
    df['Water Temp'] = predictions[:, 1]
    df.to_csv(database_file, mode='a', header=False)
    return df

#main function that calls all other functions
def main():
    data = get_data()
    df = match_headers(data, 'json_model.config')
    predict_and_append(df, 'model.pkl', 'database.csv')

if __name__ == '__main__':
    main()
