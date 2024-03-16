# lor.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import sys

def train_logistic_regression(data_path, test_size=0.2, random_state=42):
    # Load the dataset
    df = pd.read_csv(data_path)

    # Extract features and target variable
    X = df.drop(columns=df.columns[-1])  # Assume last column is target variable
    y = df[df.columns[-1]]  # Target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Creating and training the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Making predictions on the test set
    y_pred = model.predict(X_test)

    # Calculating accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Print accuracy
    print("Accuracy:", accuracy)

    # Print some predicted and actual labels for verification
    print("Sample Predicted Labels:", y_pred[:5])
    print("Sample Actual Labels:", y_test[:5])

    return model

if __name__ == "__main__":
    # Check if dataset path is provided as argument
    if len(sys.argv) != 2:
        print("Usage: python lor.py <dataset_path>")
        sys.exit(1)

    # Get dataset path from command line argument
    data_path = sys.argv[1]

    # Train logistic regression model
    model = train_logistic_regression(data_path)

