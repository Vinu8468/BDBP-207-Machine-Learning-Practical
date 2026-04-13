# Implement a polynomial kernel
# K(a, b) = a[0] ** 2 * b[0] ** 2 + 2 * a[0] * b[0] * a[1] * b[1] + a[1] ** 2 * b[1] ** 2.Apply
# this kernel function and evaluate the output for the same x1 and x2
# values.Notice that the result is the same in both scenarios demonstrating
# the power of kernel trick.
def polynomial_kernel(a,b):
    return a[0]**2*b[0]**2 + 2*a[0]*b[0]*a[1]*b[1] + a[1]**2*b[1]**2

data =[[3,10],[6,10]]
poly = polynomial_kernel(data[0],data[1])
print(poly)
