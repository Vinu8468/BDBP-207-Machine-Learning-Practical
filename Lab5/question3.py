# Compute the derivative of a sigmoid function and visualize it
import numpy as np
import matplotlib.pyplot as plt

X=np.random.randint(-10,10,1000)
X=np.sort(X)
Y=[(1/(1+np.exp(-x))) for x in X]

Dir=[np.exp(-x)/((1+np.exp(-x))**2) for x in X]
plt.plot(X,Dir)
plt.xlabel("x")
plt.ylabel("Sigmoid Derivative")
plt.title("Derivative of Sigmoid Function")
plt.show()
