# Compute SONAR classification results with and without data
# pre-processing (data normalization). Perform data pre-processing
# with your implementation and with scikit-learn methods and compare
# the results.

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


sonar = fetch_openml(name='sonar', version=1, as_frame=False)

X = sonar.data
y = sonar.target

# print(X.shape)

# textual data to 0s and 1s with label Encoder
label = LabelEncoder()
y = label.fit_transform(y)

KFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


def manual_normalize(X_train, X_val):
        # compute feature-wise min and max from training data
        min_val = np.min(X_train,axis=0)
        max_val = np.max(X_train,axis=0)

        # X_train_norm = (X_train - min_val) / (max_val - min_val)
        # X_val_norm = (X_val - min_val) / (max_val - min_val)
        # this won't account for the 0 in denominator cuz it can go to zero

        range_val = max_val - min_val
        range_val[range_val == 0] = 1 # this is to prevent

        # min-max normalization
        X_train_norm = (X_train - min_val) / range_val
        X_val_norm = (X_val - min_val) / range_val

        return X_train_norm, X_val_norm


def the_main_part(mode="none"):
    accuracies=[]
    for train_index,val_index in KFold.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        if mode=="manual": #calling the function predefined
            X_train, X_val = manual_normalize(X_train, X_val)
        elif mode=="sklearn":
            min_max_scaler = MinMaxScaler()
            X_train = min_max_scaler.fit_transform(X_train)
            X_val = min_max_scaler.transform(X_val)
        elif mode=="none":
            X_train = X_train
            X_val = X_val

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        accuracies.append(accuracy)
    accuracies=np.array(accuracies)

    print("")
    print(mode.capitalize()) #printing title
    print("accuracy per fold",accuracies)
    print("Accuracy: ",np.mean(accuracies))
    print("Standard Deviation: ",np.std(accuracies))

the_main_part("manual")
the_main_part("sklearn")
the_main_part("none")

print("done")

# Manual
'''accuracy per fold [0.80952381 0.61904762 0.76190476 0.85714286 0.66666667 0.80952381
 0.9047619  0.9047619  0.7        0.85      ]
Accuracy:  0.7883333333333333
Standard Deviation:  0.09407895800940853'''

# Sklearn
'''accuracy per fold [0.80952381 0.61904762 0.76190476 0.85714286 0.66666667 0.80952381
 0.9047619  0.9047619  0.7        0.85      ]
Accuracy:  0.7883333333333333
Standard Deviation:  0.09407895800940853'''

# None
'''accuracy per fold [0.76190476 0.61904762 0.80952381 0.9047619  0.66666667 0.80952381
 0.85714286 0.80952381 0.75       0.8       ]
Accuracy:  0.7788095238095238
Standard Deviation:  0.08030871556635909'''


