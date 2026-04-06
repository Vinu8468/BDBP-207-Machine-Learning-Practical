# Implement bagging regressor without using scikit-learn
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor

data = load_diabetes()
X = data.data
y = data.target

# lets set the bagging parameter
# the number of models
n_estimators = 10

models= []

for i in range(n_estimators):
    indices = np.random.choice(len(X), len(X), replace=True)
    X_sample = X[indices]
    y_sample = y[indices]

    model = DecisionTreeRegressor(max_depth=5)
    model.fit(X_sample, y_sample)

    models.append(model)

def predict(X):
    preds=[]
    for model in models:
        pred = model.predict(X)
        preds.append(pred)
    return np.mean(preds,axis=0)

y_pred =predict(X)
mse =np.mean((y_pred-y)**2)
print("Mean Squared Error (MSE) =", mse)
