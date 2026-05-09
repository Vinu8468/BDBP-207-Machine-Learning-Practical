# f(x, y) = 4x + 2y - x^2 - 3y^2
# Find the gradient. Use that to find critical points, (x, y) that makes gradient 0
# Use the Eigenvalues of the Hessian at the point to determine whether the
# critical point is a minimum, maximum or neither
import sympy as sp

x,y = sp.symbols('x,y')
f = 4*x + 2*y - x**2 - 3*y**2

# list of partial derivative of f wrt x and y is gradient
fx = sp.diff(f,x)
fy = sp.diff(f,y)

gradient = [fx,fy]
# now to find critical points
crit_points = sp.solve([fx,fy],(x,y))
print(f"crit_points = {crit_points}")

H = sp.hessian(f,(x,y))
print(f'The Hessian of the function {f} is : ')
print(H)

print(f'Eigen values of the Hessian are : {H.eigenvals()}')

eigenvalues = list(H.eigenvals().keys())
if all(ev >0 for ev in eigenvalues):
    print("Critical point is a local Minimum")
elif all(ev <0 for ev in eigenvalues):
    print("Critical point is a local Maximum")
else:
    print("Critical point is not a local minimum or a local maximum but a saddle")
