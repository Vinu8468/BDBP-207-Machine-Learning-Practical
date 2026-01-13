
# Covariance Matrix Computation using Matrix Multiplication
# Given a design / feature matrix X with:
# - n samples (rows)
# - d features (columns)
# We compute the covariance matrix using the formula:
#     Cov(X) = (1 / (n - 1)) * (Xcᵀ · Xc)
# where:
# - Xc is the mean-centered data matrix
# - Xcᵀ is the transpose of Xc


# Input feature matrix

X = [
    [1, 0, 2],
    [0, 1, 1],
    [2, 1, 0],
    [1, 1, 1],
    [0, 2, 1]
]

n = len(X)          #rows
d = len(X[0])       #columns

#Computing column-wise mean

means = []
for j in range(d):
    column_sum = 0
    for i in range(n):
        column_sum += X[i][j]
    means.append(column_sum / n)

print("Column-wise means:")
print(means)

# Mean-center the matrix (Xc = X - mean)

Xc = []
for row in X:
    centered_row = []
    for j in range(d):
        centered_row.append(row[j] - means[j])
    Xc.append(centered_row)

print("\nMean-centered matrix Xc:")
print(Xc)

#Transpose of Xc (XcT)

Xc_T = list(zip(*Xc))

print("\nTranspose of Xc (XcT):")
print(Xc_T)

# Step 4: Matrix multiplication (XcT · Xc)

XtX = []
for i in range(d):
    row = []
    for j in range(d):
        s = 0
        for k in range(n):
            s += Xc_T[i][k] * Xc[k][j]
        row.append(s)
    XtX.append(row)

print("\nMatrix multiplication Xcᵀ · Xc:")
print(XtX)


# Covariance matrix

covariance_matrix = []
for row in XtX:
    covariance_matrix.append([value / (n - 1) for value in row])

print("\nCovariance Matrix:")
print(covariance_matrix)

