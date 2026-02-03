# Compute the derivative of a sigmoid function and visualize it
import numpy as np
import matplotlib.pyplot as plt

X=np.random.randint(-10,10,100000)
X=np.sort(X)

Y=[(1/(1+np.exp(-x))) for x in X]
Dir=[np.exp(-x)/((1+np.exp(-x))**2) for x in X]

plt.figure() # this will enable to plot both the curves on a single plot.
plt.plot(X,Y, label="Sigmoid Function")
plt.plot(X,Dir,label="Derivative of Sigmoid Function")
plt.xlabel("X values ")
plt.ylabel("Y values")
plt.title("Sigmoid Function and its Derivative")
plt.legend() # this will display the label on the left hand side.
plt.show()

