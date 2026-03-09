# Use breast_cancer.csv and use scikit learn methods,
# OrdinalEncoder, OneHotEncoder(sparse=False),
# LabelEncoder to implement complete Logistic Regression
# Model.
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load data
columns =["age","menopause","tumor_size","inv_nodes","node_caps",
           "deg_malig","breast","breast_quad","irradiat","class"]
cancer = pd.read_csv("breast-cancer.csv",header=None,names=columns)

# missing values
cancer.replace('?',np.nan,inplace=True)
cancer.fillna(cancer.mode().iloc[0],inplace=True)

# feature and target
X= cancer.drop(columns="class")
y = cancer["class"]

# label encoding target
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# ORDINAL ENCODING

ord = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
X_ord_train = ord.fit_transform(X_train)
X_ord_test = ord.transform(X_test)

logistic_ord= LogisticRegression()
logistic_ord.fit(X_ord_train,y_train)
pred_ord = logistic_ord.predict(X_ord_test)
accuracy_ordinal = accuracy_score(y_test, pred_ord)
print("accuracy(ordinal encoding)",accuracy_ordinal)

# ONE HOT ENCODING
onehot = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
X_one_train = onehot.fit_transform(X_train)
X_one_test = onehot.transform(X_test)
logistic = LogisticRegression()
logistic.fit(X_one_train,y_train)
pred_one = logistic.predict(X_one_test)
accuracy_one = accuracy_score(y_test, pred_one)
print("accuracy(onehot encoding)",accuracy_one)

