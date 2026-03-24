# Implement Gradient Boost Regression and Classification using
# scikit-learn. Use the Boston housing dataset from the ISLP
# package for the regression problem and weekly dataset from the
# ISLP package and use Direction as the target variable for the
# classification.

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from ISLP import load_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error,accuracy_score
# Gradient Boosting Regressor
# load dataset
boston = load_data('Boston')

#featurs and target
X= boston.drop("medv",axis=1)
y = boston["medv"]

# features and target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123
)

# model
gradreg=GradientBoostingRegressor(random_state=123,
                                    n_estimators=100,
                                    learning_rate=0.1,
                                    max_depth=3)
# train
gradreg.fit(X_train, y_train)

# predict
y_pred = gradreg.predict(X_test)

# evalution
print("R2 Score", r2_score(y_test, y_pred))
print("Mean Squared Error", mean_squared_error(y_test, y_pred))

# Gradient Boosting Classifier
weekly = load_data('Weekly')

# features and target
X= weekly.drop(["Direction"],axis=1)
y= weekly["Direction"]

# convert target to binary
y=y.map({"Down":0,"Up":1})

# train_test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123
)

# model
gradclas = GradientBoostingClassifier(
    random_state=123,
    max_depth=3,
    learning_rate=0.1,
    n_estimators=100)

# train
gradclas.fit(X_train, y_train)

# predict
y_pred = gradclas.predict(X_test)

# evalution
print("Accuracy Score", accuracy_score(y_test, y_pred))

