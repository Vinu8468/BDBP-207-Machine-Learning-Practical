# Perform 10-fold cross validation for SONAR dataset in scikit-learn
# using logistic regression. SONAR dataset is a binary classification
# problem with target variables as Metal or Rock. i.e. signals are from
# metal or rock.
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

sonar = fetch_openml(name='sonar', version=1, as_frame=False)

X = sonar.data
y = sonar.target

# print(X.shape)
# since the target is labeled as rock and metal .. we should label it with label encoder
label = LabelEncoder()
y = label.fit_transform(y)

KFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

accuracies=[]
for train_idx , val_idx in KFold.split(X,y): # this will return the indexes for the training and validating set
    X_train,X_val = X[train_idx],X[val_idx] # take the corresponding values
    y_train,y_val = y[train_idx],y[val_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred=model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    accuracies.append(acc)

accuracies = np.array(accuracies)

print("Accuracy for each fold is",accuracies)
print("Mean Accuracy is",accuracies.mean())
print("Standard Deviation is",accuracies.std())