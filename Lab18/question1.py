# Consider the following dataset #data.csv.
# Implement the RBF kernel. Check if RBF kernel
# separates the data well and compare it with the Polynomial Kernel.
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import LabelEncoder
# load the data
data = pd.read_csv("data.csv")

# x= data.drop(columns ="Label") # generally use this if there are way too many features to type
x = data[["x1","x2"]]
y = data["Label"]

# since blue and red cannot be interpreted..
# y= y.map({"Blue":0,"Red":1})
le = LabelEncoder()
y = le.fit_transform(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

# radial basis function
rbf_kernal = SVC(kernel="rbf", C=1, gamma=0.1)
rbf_kernal.fit(x_train, y_train)

y_pred_rbf = rbf_kernal.predict(x_test)
accuracy = accuracy_score(y_test, y_pred_rbf)
confusion_matrix_rbf = confusion_matrix(y_test, y_pred_rbf)
print(accuracy)
# 0.6
print(confusion_matrix_rbf)
# [[1 2]
#  [0 2]]

# polynomial function 
poly_kernal = SVC(kernel="poly",degree=3, C=1)
poly_kernal.fit(x_train, y_train)

y_pred_poly = poly_kernal.predict(x_test)
accuracy = accuracy_score(y_test, y_pred_poly)
confusion_matrix_p = confusion_matrix(y_test, y_pred_poly)
print(accuracy)
# 0.2
print(confusion_matrix_p)
# [[1 2]
#  [2 0]]
