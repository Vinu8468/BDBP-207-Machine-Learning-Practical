# Implement logistic regression using scikit-learn for the breast cancer dataset

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score # this is the R2 equivalent of the

cancer =load_breast_cancer()
X = cancer.data
Y = cancer.target

# training and test split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3,random_state=500)

# training the model
LogR=LogisticRegression(max_iter=10000)# setting this to prevent overflow error
LogR.fit(X_train,Y_train)

#predictions of y values using the model
Y_pred=LogR.predict(X_test)

#checking the model accuracy.
Accuracy = accuracy_score(Y_test, Y_pred)

print("Accuracy:",Accuracy)

