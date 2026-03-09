# Implement a regression decision tree algorithm using scikit-learn for the simulated dataset.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,r2_score

data = pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
X= data.drop(columns=["disease_score","disease_score_fluct"]).values
y= data["disease_score"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# decision tree doesn't need any standard scaling
Tree = DecisionTreeRegressor(max_depth=5)
Tree.fit(X_train, y_train) # fit the model
y_pred = Tree.predict(X_test) # find the predicted values

r2_score = r2_score(y_test, y_pred) # test it against the ground truth
mse_score = mean_squared_error(y_test, y_pred)
print(f"R2 score: {r2_score}")
print(f"MSE score: {mse_score}")
