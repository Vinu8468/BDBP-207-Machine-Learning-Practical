# Compute the derivative of a sigmoid function and visualize it
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s =sigmoid(x)
    return s*(1-s)

X = np.linspace(-10, 10, 1000)
Y = sigmoid(X)
Dir = sigmoid_derivative(X)

plt.plot(X, Dir)
plt.xlabel("x")
plt.ylabel("Sigmoid Derivative")
plt.title("Derivative of Sigmoid Function")
plt.show()
