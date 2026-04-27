# For the heart.csv dataset, build a logistic regression classifier
# to predict the risk of heart disease.  Vary the threshold to generate
# multiple confusion matrices.  Implement a python code to calculate the
# following metrics
#    Accuracy
#    Precision
#    Sensitivity
#    Specificity
#    F1-score
#    Plot the ROC curve
#    AUC

# load all the important packages and libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import sys

df = pd.read_csv("Heart.csv",index_col=0)
print(df.head())
print(df.describe())
print(df.shape)
print(df.info()) # this was important because it told me that there were null values in few columns

# everything except column AHD is feature
X = df.drop(columns="AHD")
y = df["AHD"] # this one is target

# since y is yes and no we should convert it to 1 and 0 so label encoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# and also one hot encode the columns of x (ChestPain and thal)
X = pd.get_dummies(X,columns = ["ChestPain","Thal"],drop_first = True)

# splitting the data into train and test (30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 123)

X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())

# standardization to prevent extrapolation of one feature over others
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

logreg = LogisticRegression(max_iter = 1000)
# training the model
logreg.fit(X_train, y_train)

# predictions now
y_pred = logreg.predict(X_test)
y_prob = logreg.predict_proba(X_test)[:,1] # this one is for all the threshold
# now we have both the predicted and actual test y values

def confusion_matrix(y_test, y_pred):
    TP =0
    FP = 0
    FN =0
    TN =0
    for i in range(len(y_test)):
        if y_test[i] == y_pred[i]:
            if y_test[i] == 1:
                TP += 1
            else:
                TN += 1
        else:
            if y_test[i] == 1:
                FN += 1
            else:
                FP += 1
    return TP,FP,FN,TN

def accuracy(y_test, y_pred):
    TP, FP, FN, TN = confusion_matrix(y_test, y_pred)
    accuracy = (TP+TN)/(TP+FP+TN+FN)
    return accuracy

def precision(y_test, y_pred):
    TP, FP, FN, TN = confusion_matrix(y_test, y_pred)
    precision = (TP)/(TP+FP)
    return precision

def recall(y_test, y_pred):
    TP, FP, FN, TN = confusion_matrix(y_test, y_pred)
    recall = (TP)/(TP+FN)
    return recall

def specificity(y_test, y_pred):
    TP, FP, FN, TN = confusion_matrix(y_test, y_pred)
    specificity = (TN)/(TN+FP)
    return specificity

def f1(y_test, y_pred):
    TP, FP, FN, TN = confusion_matrix(y_test, y_pred)
    f1 = 2*TP/(2*TP+FP+FN)
    return f1

TP, FP, FN, TN = confusion_matrix(y_test, y_pred)
print(f"TP : {TP}\nTN : {TN}\n FP : {FP} \n FN : {FN}")
print(f"Accuracy of model is : {accuracy(y_test, y_pred)}")
print(f"Precision of model is :{precision(y_test, y_pred)}")
print(f"Recall of model is :{recall(y_test, y_pred)}")
print(f"Specificity of model is :{specificity(y_test, y_pred)}")
print(f"F1 score of model is :{f1(y_test, y_pred)}")

thresolds = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] # this is to calculate tpr and fpr as the threshold increases

tpr_list = []
fpr_list = []
f1_list = []
accuracy_list = []

for thresh in thresolds:
    y_pred_thresh = (y_prob >= thresh).astype(int)

    TP, FP, FN, TN = confusion_matrix(y_test, y_pred_thresh)
    acc = (TP+TN)/(TP+FP+FN+TN) if (TP+FP+FN+TN) > 0 else 0
    prec = TP/(TP+FP) if (TP+FP) > 0 else 0
    rec = TP/(TP+FN) if (TP+FN) > 0 else 0
    spec = TN/(TN+FP) if (TN+FP) > 0 else 0

    fpr = FP/(FP+TN) if (FP+TN) > 0 else 0

    tpr_list.append(rec)
    fpr_list.append(fpr)
    f1_list.append(f1)
    accuracy_list.append(acc)

# now ROC curves with few threshold values
plt.figure()
plt.plot(fpr_list, tpr_list)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("ROC Curve")
plt.show()

# for auc .. np.trapz can be used but we need sorted fpr and tpr
sorted_pairs = sorted(zip(fpr_list, tpr_list))
fpr_sorted, tpr_sorted = zip(*sorted_pairs)

auc = np.trapz(tpr_sorted, fpr_sorted)
print("AUC of the model is :", auc)



