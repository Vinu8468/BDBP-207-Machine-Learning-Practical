'''Implement a linear regression model using scikit-learn for the simulated dataset
- simulated_data_multiple_linear_regression_for_ML.csv  - to predict the “disease_score_fluct” from
multiple clinical parameters. '''

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def main():
    # loading the data set
    df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")


    print(df.head())

    X = df.drop(columns=["disease_score", "disease_score_fluct"])

    y = df["disease_score_fluct"]


    print(X.head())
    print(y.head())

    #train test splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=999
    )

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

    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    print(f"R2 Score: {r2:.4f}")



if __name__ == "__main__":
    main()
