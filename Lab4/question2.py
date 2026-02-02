# Use your implementation and train ML models for
# both californiahousing and simulated datasets and
# compare your results with the scikit-learn models.

from question1 import LinearRegressionVin # importing the question1 package that has the model made from scratch.
import pandas as pd
from sklearn.linear_model import LinearRegression # the built in package
from sklearn.metrics import r2_score # validataion package
from sklearn.model_selection import train_test_split # for slitting data into training and testing dataset.
from sklearn.datasets import fetch_california_housing # trying with california housing

# Loading the data
df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")

# Features and target (both X and Y)
X = df.drop(columns=["disease_score", "disease_score_fluct"]).values
Y = df["disease_score_fluct"].values.reshape(-1, 1)

# Train-test split (70% train, 30% test) this will be common data for both the from-scratch model and built-in model
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.3, random_state=999)

# Training the Scratch model with the training data set.
vin_version = LinearRegressionVin(alpha=0.001, iterations=10000) # this is the learning rate.
vin_version.fit(X_train,Y_train)

# Predict on test data
Y_test_pred_vin = vin_version.predict(X_test)

# Train scikit-learn model
sk_version = LinearRegression() # this will train on the same training data set.
sk_version.fit(X_train,Y_train)

# Predict on test data
Y_test_pred_sk = sk_version.predict(X_test)

# Compare R2 scores on test set
print("R2 Score (Gradient Descent(from scratch) with test split):", r2_score(Y_test,Y_test_pred_vin))
print("R2 Score (Scikit-learn) with test split):   ", r2_score(Y_test,Y_test_pred_sk))

# this is for disease_score_fluct
# R2 Score (Gradient Descent(from scratch) with test split): 0.571794069069699
# R2 Score (Scikit-learn) with test split):    0.5713672456666494

# this is for disease_score
# R2 Score (Gradient Descent(from scratch) with test split): 0.9999985650446537
# R2 Score (Scikit-learn) with test split):    0.9999999999697619

# now for the california housing
cali_house=fetch_california_housing()

X = cali_house.data
Y = cali_house.target.reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=999)
# similar 70% train and 30% test


# Train your Gradient Descent model
vin_version_ch = LinearRegressionVin(alpha=0.001, iterations=1000)
vin_version_ch.fit(X_train, Y_train)

# Predict on test data
Y_test_pred_vin = vin_version_ch.predict(X_test)

# Train scikit-learn model
sk_version_ch = LinearRegression()
sk_version_ch.fit(X_train, Y_train)

# Predict on test data
Y_test_pred_sk = sk_version_ch.predict(X_test)

# Compare R2 scores on test set
print("California Housing R2 (Gradient Descent from scratch):", r2_score(Y_test, Y_test_pred_vin))
print("California Housing R2 (Scikit-learn):   ", r2_score(Y_test, Y_test_pred_sk))

# California Housing R2 (Gradient Descent from scratch): 0.6262809460987274
# California Housing R2 (Scikit-learn):    0.6262524847284642