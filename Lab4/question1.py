# #Implement gradient descent algorithm from scratch using Python
# import sys
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

#copied from Lab3/question3

# # Use the above simulated CSV file and implement the following from scratch in Python
# # Step 1: Read the simulated CSV data
# df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
#
# # Print first few rows just to confirm data is loaded
# print(df.head())
#
# # Step 2: Create feature matrix X and target vector Y
#
# # X contains all independent variables
# X = df.drop(columns=["disease_score", "disease_score_fluct"]).values
#
# # Y contains the output we want to predict
# Y = df["disease_score_fluct"].values.reshape(-1, 1)
#
# # Number of training samples
# m = X.shape[0]
#
# # Add a column of ones to X for the bias term
# X = np.c_[np.ones((m, 1)), X]
#
# # Step 3: Hypothesis function
# # This gives predicted Y values using current theta
# def hypothesis(X, theta):
#     return X.dot(theta)
#
# # Step 4: Cost function
# # This tells us how wrong our predictions are
# def cost_function(theta, X, Y):
#     predictions = hypothesis(X, theta)
#     errors = predictions - Y
#     cost = (1 /(2*m)) * np.sum(errors ** 2)
#     return cost
#
# # Step 5: Gradient calculation
# # This decides how theta should change
# def compute_gradient(theta, X, Y):
#     predictions = hypothesis(X, theta)
#     gradient = (1 / m) * X.T.dot(predictions - Y)
#     return gradient
#
# # Step 6: Gradient descent algorithm
# def gradient_descent(theta, alpha, X, Y, iterations):
#     cost_history = []
#
#     for i in range(iterations):
#         cost = cost_function(theta, X, Y)
#         cost_history.append(cost)
#
#         gradient = compute_gradient(theta, X, Y)
#         theta = theta - alpha * gradient
#
#         # Print cost occasionally to see learning progress
#         if i % 100 == 0:
#             print("Iteration", i, "Cost:", cost)
#
#         # Safety check to stop if values explode
#         if np.any(np.isnan(theta)) or np.any(np.isinf(theta)):
#             print("Numerical issue detected. Stopping training.")
#             break
#
#     return theta, cost_history
#
# # Step 7: Initialize parameters
# n_features = X.shape[1]
# theta = np.zeros((n_features, 1))
#
# alpha = 0.0001
# iterations = 10000
#
# # Train the model
# theta_final, cost_history = gradient_descent(theta, alpha, X, Y, iterations)


# Designing a package/module for reusablity in other file
import numpy as np
class LinearRegressionVin:
    # Multiple Linear Regression using Gradient Descent
    # Implemented from scratch
    def __init__(self, alpha=0.001, iterations=1000, normalize=True): # setting the default value.
        self.alpha = alpha
        self.iterations = iterations
        self.normalize = normalize
        self.theta = None
        self.cost_history = []
        self.mean = None
        self.std = None

    def _normalize(self, X): # normalising the data for better fit
        self.mean = np.mean(X,axis=0)
        self.std = np.std(X,axis=0)
        return (X - self.mean) / self.std

    def _add_bias(self, X): # the 1st column of ones
        m = X.shape[0]
        return np.c_[np.ones((m, 1)), X]

    def _hypothesis(self, X):
        return X.dot(self.theta)

    def _cost(self, X, Y):
        m = len(Y)
        predictions = self._hypothesis(X)
        return (1 / (2 * m)) * np.sum((predictions - Y) ** 2) # predictions - Y is the error.

    def _gradient(self, X, Y):
        m = len(Y)
        return (1 / m) * X.T.dot(self._hypothesis(X) - Y)

    def fit(self, X, Y): # the gradient descent fit model
        if self.normalize:
            X = self._normalize(X)

        X = self._add_bias(X)

        m, n = X.shape
        self.theta = np.zeros((n, 1))

        for _ in range(self.iterations):
            self.theta = self.theta - self.alpha * self._gradient(X, Y) # this does the main task.
            self.cost_history.append(self._cost(X, Y)) # calling the func to calculate cost

        return self

    def predict(self, X):
        if self.normalize:
            X = (X - self.mean) / self.std

        X = self._add_bias(X)
        return self._hypothesis(X) # this will return the predicted y value


