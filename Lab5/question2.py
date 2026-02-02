# Implement sigmoid function in python and visualize it
import numpy as np
from matplotlib import pyplot as plt

X=np.random.randint(-10,10,100000)
X=np.sort(X)
Y=[(1/(1+np.exp(-x))) for x in X]

# Y=[]
# for x in X:
#     yval=(1/(1+np.exp(-x)))
#     Y.append(yval)

plt.plot(X,Y)
plt.xlabel('Range of X')
plt.ylabel('Range of Y')
plt.title("Sigmoid Function")
plt.show()

