# implement stochastic gradient descent from scratch
# load the important packages and data (we'll be using california housing for now)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score,mean_squared_error

# load the data
data = fetch_california_housing()
X = data.data
y = data.target
# split into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=123)

# standardization done after splitting
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

m=X_train.shape[0]
n=X_test.shape[0]
X_train = np.c_[np.ones(m),X_train]
X_test = np.c_[np.ones(n),X_test]

# The hypothesis function
def hypothesis(theta,X):
    return np.dot(X,theta)

# The cost function
def compute_cost(theta,X,y):
    m = len(y)
    prediction = hypothesis(theta,X)
    error = prediction -y
    cost = (1/(2*m))*np.dot(error.T,error)
    return cost

# the stochastic gradient
def stochastic_gradient_descent(X,y,theta,alpha,epochs):
    cost_history = []
    m = len(y)
    for epoch in range(epochs):
        indices = np.random.permutation(m) # this is the permutation of index involved in the entire strech of datasets
        X_temp = X[indices] # index to value
        y_temp = y[indices]
        for i in range(m):
            x_i = X_temp[i] # a sample for each index is chosen
            y_i = y_temp[i]
            prediction = hypothesis(theta,x_i)
            error = prediction - y_i
            gradient = error*x_i # no requirement of mean or sum because there is only one sample
            theta = theta -alpha*gradient
        cost = compute_cost(theta,X,y)
        cost_history.append(cost)
        if epoch % 10 == 0:
            print("epoch:",epoch,"cost:",cost)
    return theta,cost_history


theta = np.zeros(X_train.shape[1])
alpha = 0.01 # initializing alpha and epochs
epochs = 50

theta, cost_history = stochastic_gradient_descent(X_train,y_train,theta,alpha,epochs)

plt.plot(cost_history)
plt.title("Cost Function")
plt.xlabel("Epochs")
plt.ylabel("Cost")
# plt.show()

# this is the testing part
y_pred = hypothesis(theta, X_test)
# getting the model r2 and mean squared error when tested against ground truth
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("R2 Score:",r2)
print("Mean Squared Error:",mse)
