import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import tkinter as tk
from tkinter import ttk
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import numpy as np

def show_clusters_seaborn(data, columns, n_centroids=2):
    extracted_data = data[columns]

    km = KMeans(n_clusters=n_centroids)
    y_km = km.fit_predict(extracted_data)

    
    data['Cluster'] = y_km

    # Set the style of seaborn
    sns.set(style="whitegrid")

    # Plot the clusters using Seaborn
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=columns[0], y=columns[1],
        hue='Cluster',
        palette=sns.color_palette("husl", n_colors=n_centroids),
        data=data,
        s=100,
        legend="full"
    )

    # Plot centroids
    plt.scatter(
        km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
        s=250, marker='*',
        c='red', edgecolor='black',
        label='centroids'
    )

    plt.legend()
    plt.xlabel(columns[0])
    plt.ylabel(columns[1])
    plt.title(f'K-Means Clustering with {n_centroids} Clusters')
    plt.show()

def linear_regression(data, feature_columns, target_column):
    X = data[feature_columns]
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   
    model = LinearRegression()
    model.fit(X_train, y_train)

    
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

def plotly_scatter(data, x_column, y_column, color_column=None, title=''):
    fig = px.scatter(data, x=x_column, y=y_column, color=color_column, title=title)
    fig.show()

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

def train_neural_network(data, feature_columns, target_column):
    X = data[feature_columns].to_numpy()
    y = data[target_column].to_numpy()

 
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)

    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)


    model = SimpleNN(input_size=len(feature_columns), hidden_size=8, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

   
    epochs = 1000
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Make predictions on the test set
    with torch.no_grad():
        predicted = model(X_test_tensor).numpy()

    # Calculate and print the mean squared error
    mse = mean_squared_error(y_test, predicted)
    print(f'Neural Network Mean Squared Error: {mse}')

def main():
    # Load dataset
    df = pd.read_csv('cars.csv')

    # Show K-Means clusters using Seaborn
    show_clusters_seaborn(df, ['cubicinches', 'weightlbs'], n_centroids=4)

    # Perform linear regression
    linear_regression(df, ['cubicinches'], 'mpg')

    # Show interactive scatter plot using Plotly
    plotly_scatter(df, 'cubicinches', 'mpg', color_column='cylinders', title='Scatter Plot with Plotly')

    # Train a simple neural network
    train_neural_network(df, ['cubicinches', 'weightlbs'], 'mpg')

    # Create a basic Tkinter GUI
    root = tk.Tk()
    root.title("Data Mining GUI")

    label = ttk.Label(root, text="Welcome to the Data Mining GUI!", font=("Helvetica", 16))
    label.pack(pady=10)

    exit_button = ttk.Button(root, text="Exit", command=root.destroy)
    exit_button.pack(pady=10)

    root.mainloop()

if __name__ == '__main__':
    main()
