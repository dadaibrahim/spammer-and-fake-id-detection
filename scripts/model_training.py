import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def train(X, y):
    """Trains and predicts dataset with a Neural Network classifier using scikit-learn."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the dataset
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define the MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, random_state=42)

    # Train the classifier
    mlp.fit(X_train, y_train)

    # Predict the test set
    y_pred = mlp.predict(X_test)

    return y_test, y_pred
