# lr.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

def load_data_and_train_model(data_path):
    # Load the dataset
    df = pd.read_csv(data_path)

    # Assuming your dataset has features and target variable columns
    X = df.drop(columns=["target_column"])  # Features
    y = df["target_column"]  # Target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creating and training the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Making predictions on the test set
    y_pred = model.predict(X_test)

    # Calculating evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print evaluation metrics
    print("Mean Squared Error:", mse)
    print("R^2 Score:", r2)

    return model
