# Try classifying classes 1 and 2 from the iris dataset with SVMs,
# with the 2 first features. Leave out 10% of each class and
# test prediction performance on these observations.
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# loading dataset
iris = load_iris()

# selecting only 2 feature
X = iris.data[:,:2]
y = iris.target

# standard scaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# train test split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=42)

rbf = SVC(kernel='rbf', C=1, gamma=0.1)
rbf.fit(x_train,y_train)

y_pred = rbf.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# 0.9333333333333333
# [[6 0 0]
#  [0 5 1]
#  [0 0 3]]
# this will give a 3x3 matrix which has true 0 1 2 on top and pred 0 1 2 on side

# lets try to convert target dataset from 0 1 2  where
# 0: Iris Setosa
# 1: Iris Versicolor
# 2: Iris Virginica
# to 0 and 1 where
# 0: Iris Setosa
# 1: Rest


y = np.where(y==0,0,1)
print(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
rbf = SVC(kernel='rbf', C=1, gamma=0.1)
rbf.fit(x_train,y_train)
y_pred = rbf.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# 1.0
# [[6 0]
#  [0 9]]