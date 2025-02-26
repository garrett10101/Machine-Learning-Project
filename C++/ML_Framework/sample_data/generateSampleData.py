import yfinance as yf
import pandas as pd
import os

def generate_stock_data(ticker, start_date, end_date, filename):
    # Download stock data with daily interval
    data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    
    # Repeat the data to increase the size
    while data.memory_usage(deep=True).sum() < 1024 * 1024:  # 100MB
        data = pd.concat([data, data])
    
    # Save to CSV to current directory
    data.to_csv(filename)
path = r"C:\Users\gad50\Desktop\Machine-Learning-Project\C++\ML_Framework\sample_data\sample_data_regression.csv"
# Example usage
generate_stock_data('AAPL', '2020-01-01', '2020-12-31', path)

import requests
import os
import pandas as pd
from PIL import Image
from io import BytesIO
import numpy as np

def download_image(url, filename):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img.save(filename)

def extract_features(image_path, size=(64, 64)):
    with Image.open(image_path) as img:
        img = img.resize(size)
        img_array = np.array(img).flatten()
        return img_array

def generate_image_data(landscape_urls, dog_urls, output_csv):
    data = []
    image_dir = "images"
    os.makedirs(image_dir, exist_ok=True)
    
    for i, url in enumerate(landscape_urls):
        filename = os.path.join(image_dir, f"landscape_{i}.jpg")
        download_image(url, filename)
        features = extract_features(filename)
        data.append([filename] + features.tolist() + ["landscape"])
    
    for i, url in enumerate(dog_urls):
        filename = os.path.join(image_dir, f"dog_{i}.jpg")
        download_image(url, filename)
        features = extract_features(filename)
        data.append([filename] + features.tolist() + ["dog"])
    
    columns = ["filename"] + [f"pixel_{i}" for i in range(len(data[0]) - 2)] + ["label"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False)

landscape_urls = [
    "https://example.com/landscape1.jpg",
    "https://example.com/landscape2.jpg",
    # Add more landscape image URLs
]

dog_urls = [
    "https://example.com/dog1.jpg",
    "https://example.com/dog2.jpg",
    # Add more dog image URLs
]

path = r"C:\Users\gad50\Desktop\Machine-Learning-Project\C++\ML_Framework\sample_data\sample_data_classifiers.csv"
generate_image_data(landscape_urls, dog_urls, path)