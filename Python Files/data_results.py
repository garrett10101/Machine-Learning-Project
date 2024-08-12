#This file takes input data and shows simple data analysis results, including heat maps, histograms, etc.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pytz

#function to show heat map
def show_heat_map(df, title):
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title(title)
    plt.show()
#function to show histogram
def show_histogram(df, column, title):
    plt.figure(figsize=(12, 10))
    sns.histplot(df[column], kde=True)
    plt.title(title)
    plt.show()
#function to show scatter plot
def show_scatter_plot(df, x, y, title):
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=df, x=x, y=y)
    plt.title(title)
    plt.show()
#function to show bar plot
def show_bar_plot(df, x, y, title):
    plt.figure(figsize=(12, 10))
    sns.barplot(data=df, x=x, y=y)
    plt.title(title)
    plt.show()
#function to show line plot
def show_line_plot(df, x, y, title):
    plt.figure(figsize=(12, 10))
    sns.lineplot(data=df, x=x, y=y)
    plt.title(title)
    plt.show()
#function to show pair plot
def show_pair_plot(df, title):
    plt.figure(figsize=(12, 10))
    sns.pairplot(df)
    plt.title(title)
    plt.show()
#function to get time series analysis
def time_series_analysis(df, column):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df[column].plot(figsize=(12, 10))
    plt.show()
#function to use PCA to reduce number of features
def pca_analysis(df, n_components):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    pca.fit(df)
    return pca
#function to create a correlation matrix
def correlation_matrix(df):
    return df.corr()
#function to create a covariance matrix
def covariance_matrix(df):
    return df.cov()

#usage example
df = pd.read_csv('Data/output_data.csv')
show_heat_map(df, 'Heat Map')