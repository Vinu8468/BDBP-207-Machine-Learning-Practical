# For a design or feature matrix,
# X=[[1,0,2],
#    [0,1,1],
#    [2,1,0],
#    [1,1,1],
#    [0,2,1]]
# Compute the covariance matrix using matrix multiplications.
# Verify your results by using numpy library operations

# number of samples: n=5
# number of features: d=3

# covariance matrix formula
# Cov(X)=1/n-1(XcT Xc)
# Xc=X-u(column - wise mean centred matrix)
# n-1=4


X=[
   [1,0,2],
   [0,1,1],
   [2,1,0],
   [1,1,1],
   [0,2,1]
   ]
n=len(X)

means = []
for i in range(len(X[0])):
    ele=0
    for j in range(len(X)):
        ele+=X[j][i]
    means.append(ele/n)

print(means)
# Mean-centered matrix Xc
Xc = []
for row in X:
    centered_row = []
    for j in range(len(row)):
        centered_row.append(row[j] - means[j])
    Xc.append(centered_row)

print(Xc)
# Transpose Xc
Xc_T = list(zip(*Xc))
# Matrix multiplication: Xc_T @ Xc
XtX = []
for i in range(len(Xc_T)):
    row = []
    for j in range(len(Xc_T)):
        s = 0
        for k in range(len(Xc)):
            s += Xc_T[i][k] * Xc[k][j]
        row.append(s)
    XtX.append(row)

print(XtX)
# Covariance matrix
cov = []
for row in XtX:
    cov.append([val / (n - 1) for val in row])

print(cov)
#verification using numpy
import numpy as np

X_np = np.array(X)

# NumPy covariance
cov_np = np.cov(X_np, rowvar=False)

print(cov_np)
