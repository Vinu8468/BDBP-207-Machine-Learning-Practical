# Build a classification model for wisconsin dataset using Ridge and Lasso classifier using scikit-learn

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=500)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Ridge (L2)
ridge_model = RidgeClassifier(alpha=0.01)
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)
# this is true ridge which gives square loss not (log loss)

print("Ridge Classifier")
print("Accuracy:", accuracy_score(y_test, ridge_pred))

# Lasso (L1)
lasso_model = LogisticRegression(
    penalty='l1',
    solver='liblinear',
    max_iter=10000)

lasso_model.fit(X_train, y_train)
lasso_pred = lasso_model.predict(X_test)

print("Lasso Classifier")
print("Accuracy:", accuracy_score(y_test, lasso_pred))


