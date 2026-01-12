# Implement california housing prediction model using
# scikit-learn - walkthro’ of
# bdbp207_californiahousing.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score


def eda(X_df, X_df_scaled):
    print("Original data summary:")
    print(X_df.describe())

    print("\nScaled data summary:")
    print(pd.DataFrame(X_df_scaled, columns=X_df.columns).describe())


def load_data():
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    return X, y

# sys.exit()
def main():
    # Load dataset
    X, y = load_data()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=999
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # EDA
    eda(X_train, X_train_scaled)

    # Model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Prediction
    y_pred = model.predict(X_test_scaled)

    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nMean Absolute Error:", mae)
    print("R² Score:", r2)


if __name__ == "__main__":
    main()
