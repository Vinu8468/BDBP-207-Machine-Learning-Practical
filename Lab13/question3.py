# Implement Random Forest algorithm for regression and classification using scikit-learn. Use diabetes and iris datasets.
# import packages
from sklearn.ensemble import RandomForestClassifier , RandomForestRegressor
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
# Random Forest regression
# loading and splitting into train and test
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# model training
regressor = RandomForestRegressor(n_estimators=100,random_state=123)
regressor.fit(X_train, y_train)

# prediction
y_pred = regressor.predict(X_test)
print("R2 score of RandomForestRegressor on Diabetes Dataset", r2_score(y_test, y_pred))
print("Mean squared error of RandomForestRegressor on Diabetes Dataset", mean_squared_error(y_test, y_pred))

# Random Forest classification
iris = load_iris()
X = iris.data
y = iris.target

# train, test splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

#training model
clf = RandomForestClassifier(n_estimators=100,random_state=123)
clf.fit(X_train, y_train)

# prediction
y_pred = clf.predict(X_test)

# reporting the accuracy
print("accuracy score of RandomForestClassifier on Iris Dataset", accuracy_score(y_test, y_pred))
