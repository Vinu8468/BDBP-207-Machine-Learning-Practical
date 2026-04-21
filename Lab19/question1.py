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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('Heart.csv', index_col=0)
# print(data.head())
# print(data.describe())
# print(data.shape)
# print(data.info())

# manual label encoding for the target
df["AHD"] = df["AHD"].map({"No": 0, "Yes": 1})

# encoding of data for chestPain and Thal
df = pd.get_dummies(df, columns=["ChestPain", "Thal"], drop_first=True)

# Features and Target
X = df.drop(columns=["AHD"], axis=1)
y = df["AHD"]

# splitting the data into train and test (30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123
)

# print(X_train.head())
# print(y_test.head())

# train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# get the probabilities
y_prob = model.predict_proba(X_test)[:, 1]
y_true = y_test.values


class MetricsEvaluatorVin:

    # confusion matrix
    def confusion_matrix(self, y_true, y_prob, threshold):
        y_pred = (y_prob >= threshold).astype(int)

        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))  # FIXED
        FP = np.sum((y_true == 0) & (y_pred == 1))  # FIXED
        FN = np.sum((y_true == 1) & (y_pred == 0))

        return TP, TN, FP, FN

    # all the metrics accuracy, precision, sensitivity, specificity and F1
    def calculate_metrics(self, TP, TN, FP, FN):
        accuracy = (TP + TN) / (TP + TN + FP + FN)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0

        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        F1 = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

        return accuracy, precision, sensitivity, specificity, F1

    # ROC curve which is taken at all the possible thresholds
    def roc_curve(self, y_true, y_prob):
        thresholds = np.linspace(0, 1, 100)  # FIXED

        TPR = []
        FPR = []

        for t in thresholds:
            TP, TN, FP, FN = self.confusion_matrix(y_true, y_prob, t)

            tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
            fpr = FP / (FP + TN) if (FP + TN) > 0 else 0

            TPR.append(tpr)
            FPR.append(fpr)

        return np.array(FPR), np.array(TPR)  # FIXED ORDER

    # plotting the roc
    def plot_roc(self, FPR, TPR):
        plt.plot(FPR, TPR)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()

    # computing auc using trapezoidal rule
    def compute_auc(self, FPR, TPR):
        return np.trapz(TPR, FPR)  # FIXED


evaluate = MetricsEvaluatorVin()
thresholds = [0.3, 0.5, 0.7]

for t in thresholds:
    TP, TN, FP, FN = evaluate.confusion_matrix(y_true, y_prob, threshold=t)

    acc, prac, sens, spec, f1 = evaluate.calculate_metrics(TP, TN, FP, FN)

    print(f"\nThreshold: {t}")
    print(f"Accuracy: {acc}")
    print(f"Precision: {prac}")
    print(f"Sensitivity: {sens}")
    print(f"Specificity: {spec}")
    print(f"F1 score: {f1}")

# ROC + AUC
FPR, TPR = evaluate.roc_curve(y_true, y_prob)
evaluate.plot_roc(FPR, TPR)

auc = evaluate.compute_auc(FPR, TPR)
print(f"AUC: {auc}")