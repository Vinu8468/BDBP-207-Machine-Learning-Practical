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