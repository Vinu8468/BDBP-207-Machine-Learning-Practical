# Implement bagging regressor and classifier using scikit-learn. Use diabetes and iris datasets.
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_diabetes
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

# Bagging regressor
# load the dataset
diabetes = load_diabetes()
X=diabetes.data
y=diabetes.target

#train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Bagging Regressor
regressor =BaggingRegressor(
    estimator=DecisionTreeRegressor(),
    n_estimators=100,
    random_state=123
)

# training
regressor.fit(X_train, y_train)

#prediction
y_pred = regressor.predict(X_test)

# scores
r2=r2_score(y_test, y_pred)
mse=mean_squared_error(y_test, y_pred)
print("R2 score for the bagging regressor is:", r2)
print("MSE score for the bagging regressor is:", mse)

# Bagging Classifier
# Load Data
iris=load_iris()
X=iris.data
y=iris.target

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Bagging Regressor
classifier=BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    random_state=123
)
# Train
classifier.fit(X_train, y_train)

# prediction
y_pred = classifier.predict(X_test)
accuracy=accuracy_score(y_test, y_pred)
print("Accuracy score for the bagging regressor is:", accuracy)