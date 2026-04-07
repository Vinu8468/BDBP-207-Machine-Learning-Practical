# Implement XGBoost classifier and regressor using scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes, load_iris
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score

# Regressor
# loading data
data = load_diabetes()
X = data.data
y = data.target

# spliting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# model
xgreg = XGBRegressor(
    learning_rate=0.1,
    n_estimators=100,
    max_depth=3,
    random_state=123
)
# training
xgreg.fit(X_train,y_train)

# predictions
y_pred = xgreg.predict(X_test)

# evaluations
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE by XGBoost Regressor on diabetes datasets: ",mse)
print("R2 by XGBoost Regressor on diabetes datasets:",r2)

#####################3##########################
print("_________________________________________________________________")
# Classifier

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

xgcla = XGBClassifier(
    learning_rate=0.1,
    n_estimators=100,
    max_depth=3,
    use_label_encoder=False,
    eval_metric="mlogloss"
)

# training
xgcla.fit(X_train,y_train)

# predict
y_pred = xgcla.predict(X_test)

# accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy with XGBoost Classifier for iris dataset: ",accuracy)
