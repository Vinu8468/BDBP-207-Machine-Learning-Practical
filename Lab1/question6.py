# Implement y = 2x1 + 3x2 + 3x3 + 4, where x1, x2 and x3 are
# three independent variables.
# Compute the gradient of y at a few points and print the values.

def y(x1,x2,x3):
    return 2*x1+3*x2+3*x3+4

# Gradient function
# here we should calculate the partial derivative
# for example in the above equation y= 2x1 + 3x2 + 3x3 + 4, dy/dx1= 2 as everything else is considered to be constant while differentiting.
def gradient():
    # Partial derivatives
    dy_dx1 = 2
    dy_dx2 = 3
    dy_dx3 = 3
    return dy_dx1, dy_dx2, dy_dx3

# Test points
points = [
    (0, 0, 0),
    (1, 2, 3),
    (-1, 4, 2)
]

# Compute and print gradient at each point
for p in points:
    grad = gradient()
    print(f"At point x1={p[0]}, x2={p[1]}, x3={p[2]} → Gradient = {grad}")
# we can infer that...
# The gradient does not depend on the point,
# Linear functions have constant slope,
# There are no stationary points since the gradient never becomes zero
