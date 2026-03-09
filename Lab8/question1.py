# Implement L2-norm and L1-norm from scratch
# import the packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load the data set
data = pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
X= data.drop(columns=["disease_score","disease_score_fluct"]).values
y= data["disease_score_fluct"].values.reshape(-1,1)
# spliting
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=123)
# standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# add bias
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# hypothesis
def hypothesis(X, theta):
    return np.dot(X, theta)

# compute cost for l2 (ridge)
# this is sse cost not mse cost
def cost_L2(theta,X,y,lambd):
    predictions = hypothesis(X, theta)
    error = predictions - y
    cost = (1/2)*np.sum(error**2)
    penalty = lambd*np.sum(theta[1:]**2)
    return cost + penalty

# this is also sse gradient
def gradient_L2(theta,X,y,lambd):
    predictions = hypothesis(X, theta)
    error = predictions - y
    grad = np.dot(X.T,error)
    reg = np.zeros_like(theta)
    reg[1:]= 2*lambd*theta[1:]
    return grad + reg
# compute cost for L1 (lasso)
def cost_L1(theta,X,y,lambd):
    predictions = hypothesis(X, theta)
    error = predictions - y
    cost = (1/2)*np.sum(error**2)
    penalty = lambd*np.sum(np.abs(theta[1:]))
    return cost + penalty

def gradient_L1(theta,X,y,lambd):
    predictions = hypothesis(X, theta)
    error = predictions - y
    grad = np.dot(X.T,error)
    reg = np.zeros_like(theta)
    reg[1:]= lambd* np.sign(theta[1:])
    return grad + reg

# a function that can choose either l1 or l2
def gradient_descent(theta, X, y, lambd, alpha, iterations, reg_type):
    cost_history = []
    for i in range(iterations):
        if reg_type == "L1":
            grad = gradient_L1(theta, X, y, lambd)
            cost = cost_L1(theta, X, y, lambd)
        elif reg_type == "L2":
            grad = gradient_L2(theta, X, y, lambd)
            cost = cost_L2(theta, X, y, lambd)
        theta = theta - alpha * grad
        cost_history.append(cost)
        if i % 100 == 0:
            print(f"{reg_type} cost after {i} iterations: {cost}")
    return theta, cost_history

alpha = 0.0001
lambd = 0.1
iterations = 1000

theta_L1 = np.zeros((X_train.shape[1],1))
theta_L2 = np.zeros((X_train.shape[1],1))
 # two separate thetas to hinder with because we might end up bothering the theta which is changed once
theta_L1, cost_L1_hist = gradient_descent(theta_L1,X_train,y_train,lambd,alpha,iterations,"L1")
theta_L2, cost_L2_hist = gradient_descent(theta_L2,X_train,y_train,lambd,alpha,iterations,"L2")
plt.plot(cost_L1_hist)
plt.plot(cost_L2_hist)
plt.show()
