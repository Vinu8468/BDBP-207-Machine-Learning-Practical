# Determine the concavity of
# f(x, y) = x^3 + 2y^3 -xy  at  (i) (0,0,  (ii) (3, 3),  (iii) (3, -3)

import sympy as sp
x,y = sp.symbols('x,y') # these are the variables

# the function..
f = x**3 +2*y**3 -x*y

# first to calculate hessian...
# first calculate the partial first derivatives of of the function with respect to x and y (fx and fy)
# then the second derivative fxx , fyy ,fxy,fyx

# in this case
# fx = 3x^2 -y
# fy = 6y^2 -x

# the second derivatives are
# fxx = 6x
# fyy = 12y
# fxy = -1
# fyx = -1

# the the hessian matrix will be
# 6x -1
# -1 12y
H = sp.hessian(f,(x,y))
print("Hessian is")
print(H)

# once hessian matrix is found .. the determinent of that..
# then substitution with x and y will give a value.. if..
# then det > 0 and fxx >0 then concave up(min)
# if det >0 and fxx<0 then concave down(max)
# else det<0 then saddle point

def classify(point):
    H_sub = H.subs({x:point[0],y:point[1]})

    # determinant
    D = H_sub.det()
    #fxx
    fxx = H_sub[0,0] # this will just borrow from the previous cal

    print(f"Point {point} ")
    print("Hessian at point")
    print(H_sub)
    print("Determinant:",D)

    if D>0:
        if fxx>0:
            print("-->Convave up (locally convex)")
        elif fxx<0:
            print("-->Convave down (locally concave)")
    elif D<0:
        print("-->Saddle point")
    else:
        print("-->Inconclusive")

# evaluate for each point

points = [(0,0),(3,3),(3,-3)]

for p in points:
    classify(p)