# Implement a linear regression model using scikit-learn for the simulated dataset -
# simulated_data_multiple_linear_regression_for_ML.csv  - to predict the “disease_score” from
# multiple clinical parameters.

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def main():
    # loading the data set
    df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")

    # head will give the first 5 rows
    print(df.head())

    X = df.drop(columns=["disease_score", "disease_score_fluct"])
    # X= df[["age","BMI","BP","blood_sugar","Gender"]] # can use this too
    y = df["disease_score"]
    # defining features (X) and target (y)... the df.drop will ommit that column and list columns
    # remove disease_score_fluct too because technically its Y

    print(X.head())
    print(y.head())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    # so if I give test_size anything more than 1.0 (which is illogical) it should technically throw error.. but it doesn't...what's happening
    # if you give test_size=3 it will consider 3 data points in the data set instead of 0.3 which will consider 30%
    # print(X_train.describe())
    # print(X_test.describe())
    # print(y_train.describe())
    # print(y_test.describe())

    scaler = StandardScaler()
    #feature scaling
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # predict values
    y_pred = model.predict(X_test_scaled)

    #model evaluation
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    # average of the squared differences between the actual values and the predicted values.
    # Quantifies overall prediction error
    # Penalizes large errors heavily (because of squaring)
    # Units are squared units of the target variable
    # Lower MSE ... better model
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    # quare root of MSE, bringing the error back to the original scale of the target variable.
    # Lower RMSE ... better predictive accuracy
    print(f"R2 Score: {r2:.4f}")
    # R2 measures the proportion of variance in the target variable explained by the model.
    # How well the model explains the data variability
    # Relative, not absolute, performance metric


if __name__ == "__main__":
    main()

