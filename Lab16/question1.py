# Write a Python program to aggregate predictions from
# multiple trees to output a final prediction for a regression
# problem.
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_diabetes
import numpy as np

data = load_diabetes()
X = data.data
y = data.target

n_estimators= 5
models=[]
for _ in range(n_estimators):
    model = DecisionTreeRegressor()
    model.fit(X,y)
    models.append(model)

def aggregate_predictions(models,X):
    predictions=[]
    for m in models:
        predictions.append(m.predict(X))
    mean_prediction= np.mean(predictions,axis=0)
    return mean_prediction

predictions = aggregate_predictions(models,X)
print("Final predictions:",predictions)
