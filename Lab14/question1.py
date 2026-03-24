# Implement Adaboost classifier using scikit-learn. Use the Iris dataset.
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# load the dataset
iris = load_iris()
X = iris.data
y = iris.target

# train - test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# ada boost classifier
classifier = AdaBoostClassifier(random_state=123,n_estimators=100, estimator=DecisionTreeClassifier(max_depth=1),learning_rate=1.0)

# train
classifier.fit(X_train, y_train)

# prediction
y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of AdaBoost classifier on test set: ", accuracy)
