# Implement a classification decision tree algorithm using scikit-learn for the sonar  dataset.

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_openml

# sonar dataset is available in fetch_openml
data = fetch_openml('sonar', version=1)
X = data.data
y = data.target

# label encoding target because its in "char" datatype.
LE = LabelEncoder()
y = LE.fit_transform(y)

# split the data into test and train
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=123)

claTree = DecisionTreeClassifier(max_depth=13) #if put more the model will start overfitting.
claTree.fit(train_X, train_y)# fit the model

y_pred = claTree.predict(test_X)

accuracy = accuracy_score(test_y, y_pred)
print(f"Accuracy: {accuracy}")