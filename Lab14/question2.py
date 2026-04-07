#Implement Adaboost classifier without using
# scikit-learn. Use the Iris dataset.
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load the dataset
iris = load_iris()
X = iris.data
y = iris.target
y = np.where(y == 0, 1, -1)
# now we have to convert the target from 0,1 and 2 to 1,0,-1 this way model
# interpret it as setosa and other or any other flower
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=123)

def AdaboostClassifier(X, y, n_estimators):
    n_samples = len(y)

    # Initialize weights
    weights = np.ones(n_samples) / n_samples

    models = []
    alphas = []

    for _ in range(n_estimators):
        # Train weak learner
        model = DecisionTreeClassifier(max_depth=1)
        model.fit(X, y, sample_weight=weights)

        predictions = model.predict(X)

        # Compute error
        error = np.sum(weights * (predictions != y))

        # Avoid division by zero
        error = max(error, 1e-10)

        # Compute alpha (model weight)
        alpha = 0.5 * np.log((1 - error) / error)

        # Update weights
        weights *= np.exp(-alpha * y * predictions)

        # Normalize
        weights /= np.sum(weights)

        models.append(model)
        alphas.append(alpha)

    return models, alphas

def predict(X,models,alphas):
    final_pred = np.zeros(X.shape[0])

    for model , alpha in zip(models,alphas):
        final_pred += alpha * model.predict(X)
    return np.sign(final_pred)

models, alphas = AdaboostClassifier(X_train,y_train, n_estimators=100)

y_pred = predict(X_test,models,alphas)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)