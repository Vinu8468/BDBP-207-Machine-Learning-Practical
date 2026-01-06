# Here is a linear model.
# y = 2x1 + 3x2 + 3x3 + 4
# The coefficients, represented as theta, is a vector given below
# Q=[2,3,3]
# There are 5 samples represented in a matrix ,X, given below
# X=[[1,0,2],
#    [0,1,1],
#    [2,1,0],
#    [1,1,1],
#    [0,2,1]]
# Compute X*Q
Q=[2,
   3,
   3]

X=[[1,0,2],
   [0,1,1],
   [2,1,0],
   [1,1,1],
   [0,2,1]]

# Q*X?
X_Q=[]

for row in X:
    s = 0
    for i in range(len(Q)):
        s += row[i] * Q[i]
    X_Q.append(s)

print(X_Q)

