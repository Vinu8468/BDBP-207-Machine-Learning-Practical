# Implement a decision regression tree algorithm
# without using sklearn tree implementation
# using the diabetes dataset

import numpy as np
from sklearn.datasets import load_diabetes

# Load dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

print(X.shape)  # (442,10)
print(y.shape)  # (442,)

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTreeRegressorScratch:

    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.root = None

    # Mean Squared Error
    def mse(self, y):
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)

    # Cost of split
    def split_cost(self, y_left, y_right):

        n = len(y_left) + len(y_right)

        cost_left = self.mse(y_left)
        cost_right = self.mse(y_right)

        weighted_cost = (len(y_left)/n)*cost_left + (len(y_right)/n)*cost_right

        return weighted_cost


    def best_split(self, x, y):

        best_feature = None
        best_threshold = None
        best_cost = float("inf")

        n_samples, n_features = x.shape

        for feature in range(n_features):

            # unique sorted values
            values = np.unique(x[:, feature])

            # compute candidate midpoints
            thresholds = (values[:-1] + values[1:]) / 2

            for threshold in thresholds:

                left_idx = x[:, feature] <= threshold
                right_idx = x[:, feature] > threshold

                if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
                    continue

                y_left = y[left_idx]
                y_right = y[right_idx]

                cost = self.split_cost(y_left, y_right)

                if cost < best_cost:
                    best_cost = cost
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold


    def build_tree(self, x, y, depth):

        # stopping condition
        if depth == self.max_depth or len(set(y)) == 1:
            leaf_value = np.mean(y)
            return Node(value=leaf_value)

        feature, threshold = self.best_split(x, y)

        if feature is None:
            return Node(value=np.mean(y))

        left_idx = x[:, feature] <= threshold
        right_idx = x[:, feature] > threshold

        left_child = self.build_tree(x[left_idx], y[left_idx], depth+1)
        right_child = self.build_tree(x[right_idx], y[right_idx], depth+1)

        return Node(feature=feature, threshold=threshold, left=left_child, right=right_child)


    def fit(self, x, y):
        self.root = self.build_tree(x, y, 0)


    def predict_sample(self, x, node):

        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self.predict_sample(x, node.left)
        else:
            return self.predict_sample(x, node.right)


    def predict(self, x):

        return np.array([self.predict_sample(i, self.root) for i in x])


# Train the tree
tree = DecisionTreeRegressorScratch(max_depth=3)
tree.fit(X, y)

# Predictions
predictions = tree.predict(X)

print("First 10 Predictions:")
print(predictions[:10])