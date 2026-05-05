# Find the eigenvalues of the given Hessian at the given point
# [12x^2   -1;  -1    2]  at (3, 1)

import sympy as sp
# the task is to find eigen values of the matrix when (x,y)=(3,1)
x,y = sp.symbols('x,y') # this is the step to define these x and y variables

H = sp.Matrix([[12*x**2,-1],[-1,2]]) # this is the matrix with the variable value. this is the Hessian.

H_sub = H.subs({x:3,y:1}) # y since there isn't any y values only x is substituted

print(H_sub)

print(f"Eigenvalues : {H_sub.eigenvals()}")

# doing it from scratch
# characteristic polynomial det(H-lambdaI)
lam =sp.symbols('lam')
H=H_sub
char_det =(H-lam*sp.eye(2)).det()

print("Character Polynomial is :")
print(char_det)
# this will result in a polynomial function

# solving manually
eigenvalues = sp.solve(char_det,lam)

print("Eigenvalues of the polynomial (from scratch) is :")
print(eigenvalues)