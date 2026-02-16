# Use validation set to do feature and model selection.
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

cali=fetch_california_housing()
X=cali.data
y=cali.target

# train and test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# val set
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=50)

# scale the data before L1 regularisation
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)
x_test_scaled = scaler.transform(x_test)

#we can do feature selection with lasso (L1 regularisation) the corresponding non relavent theta will do to 0
lassso= LassoCV(cv=5, random_state=50)
lassso.fit(x_train_scaled, y_train)

# choose only the non-zero coefficient features
selected_features= np.where(lassso.coef_ != 0)[0]

X_train_sel =x_train_scaled[:,selected_features]
X_val_sel =x_val_scaled[:,selected_features]
X_test_sel =x_test_scaled[:,selected_features]

print("selected features:", len(selected_features))

#model selection

models={
    "linear regression":LinearRegression(),
    "Decision Tree" : DecisionTreeRegressor(random_state=50),
    "Random Forest" : RandomForestRegressor(n_estimators=100,random_state=50)
}

results={}

# once the models are stored in a dictionary ... we pass the train data and test the model once trained and check their
# Mean Square error and R2

for name, model in models.items():
    model.fit(X_train_sel, y_train)
    y_pred = model.predict(X_val_sel)

    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    results[name] = {"MSE":mse,"R2":r2}
    print(f"{name}:MSE: {mse:.3f}, R2: {r2:.3f}")

print(results)


