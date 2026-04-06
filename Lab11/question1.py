# Implement decision tree classifier without using scikit-learn
# using the iris dataset. Fetch the iris dataset from scikit-learn library.

from sklearn.datasets import load_iris
from collections import Counter
import numpy as np

# entropy will be calculated by - summation of -p log (p)
def Entropy(labels):
    counts = Counter(labels)
    entropy = 0
    length = len(labels)
    for count in counts.values():
        p = count / length
        entropy -= p * np.log2(p)
    return entropy

# information gain parent entropy - child entropy
def information_gain(parent, left, right):
    parent_entropy = Entropy(parent)
    n = len(parent)
    n_left = len(left)
    n_right = len(right)

    # avoid empty splits
    if n_left == 0 or n_right == 0:
        return 0

    child_entropy = (n_left/n) * Entropy(left) + (n_right/n) * Entropy(right)

    return parent_entropy - child_entropy

# spliting the data with split value and split dimension (a.k.a feature and value)
def split(X, y, feature, threshold):
    left_mask = X[:, feature] <= threshold
    right_mask = X[:, feature] > threshold
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

# best split value
def best_split(X, y):
    best_feature = None
    best_threshold = None
    best_gain = -1
    n_features = X.shape[1]

    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])

        for t in thresholds:
            X_left, y_left, X_right, y_right = split(X, y, feature, t)

            gain = information_gain(y, y_left, y_right)

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = t

    return best_feature, best_threshold

#node
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# build tree
def build_tree(X, y, depth=0, max_depth=4):
    # if pure node
    if len(np.unique(y)) == 1:
        return Node(value=y[0])
    # max depth
    if depth >= max_depth:
        return Node(value=Counter(y).most_common(1)[0][0])
    feature, threshold = best_split(X, y)

    if feature is None:
        return Node(value=Counter(y).most_common(1)[0][0])

    X_left, y_left, X_right, y_right = split(X, y, feature, threshold)

    left = build_tree(X_left, y_left, depth + 1, max_depth)
    right = build_tree(X_right, y_right, depth + 1, max_depth)

    return Node(feature, threshold, left, right)

# prediction
def predict_one(node, x):
    if node.value is not None:
        return node.value

    if x[node.feature] <= node.threshold:
        return predict_one(node.left, x)
    else:
        return predict_one(node.right, x)

def predict(tree, X):
    return np.array([predict_one(tree, x) for x in X])

#loading data
iris = load_iris()
X = iris.data

# converting numeric labels to original names
y = np.array([iris.target_names[i] for i in iris.target])

feature_names = iris.feature_names

tree = build_tree(X, y, max_depth=4)

y_pred = predict(tree, X)

accuracy = np.mean(y_pred == y)
print("Accuracy =", accuracy)

# printing tree
def print_tree(node, depth=0):
    if node.value is not None:
        print("  " * depth + f"Leaf -> {node.value}")
        return

    print("  " * depth + f"{feature_names[node.feature]} <= {node.threshold:.2f}")
    print_tree(node.left, depth + 1)

    print("  " * depth + f"{feature_names[node.feature]} > {node.threshold:.2f}")
    print_tree(node.right, depth + 1)

print("\nDecision Tree:\n")
print_tree(tree)