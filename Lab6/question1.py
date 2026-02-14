# K-fold cross validation. Implement for K = 10. Implement from scratch, then, use scikit-learn methods.
# 10 fold cross validation

from sklearn.datasets import load_iris # this is a flower data with 150 entries
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
# iris = load_iris() #This is small dataset.. so digits dataset can also be tried.
# X=iris.data
# y=iris.target
digits=load_digits()
X=digits.data
y=digits.target
K = 10 # k is set to 10. so its 10 fold cross validation.
# print(X.shape) # this gives(150, 4) so select 150 by splicing for iris 
n_samples = X.shape[0] #This is 150 a.k.a sample size.


# Shuffle data
indices = np.random.permutation(n_samples) # this is to get random values till 150. and no repeated values.
fold_size = n_samples // K # since k is 10 the fold size becomes 15

accuracies = []

for k in range(K):
    print(f"Fold :{k+1}")

    start = k * fold_size # this is 0 to 15 for first iteration, 15 to 30 and so on
    end = start + fold_size
    test_idx = indices[start:end] # this will give the 15 indices randomly but the next will not be same
    # so the test_idx will be for example indices[0:15]
    # suffling the index and then choosing the 15 index from that


    train_idx = np.concatenate((indices[:start], indices[end:])) # this gives the training data set which will be everything except what was given to test data.. so everything in 150 except the 15 which will be used for test.
    # take the corresponding datapoints of the indices
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Training model using Logistic Regression from sklearn
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # now we test with the accurary.
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    accuracies.append(acc)
    print("Accuracy:", acc)

# Final result
print("Average Accuracy (10-fold CV):", np.mean(accuracies))

