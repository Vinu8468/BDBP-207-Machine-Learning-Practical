# Compute the dot product of two vectors, x and y given below
# x = [2  1  2]T and y = [1  2  2]T . What is the meaning of the
# dot product of two vectors? Illustrate that with your own example.
def Dotproduct(X,Y):
    dotproduct = 0
    for x in range(len(X)):
        dotproduct += X[x] * Y[x]
    return dotproduct

Xval=[2,1,2]
Yval=[1,2,2]
dot=Dotproduct(Xval,Yval)
print(f"Dotproduct: {dot}")

# Dot product of two vectors signifies the direction of the 2 vectors
# if the dot product is positive.. the vectors are pointing in the same direction.
# if its negative .. the vectors are pointing in the opposite direction.
# if its zero... then the vectors are orthogonal/perpendicular to each other.

# another interpretation is it gives the magnitude of projection of one vector onto another(its scaled by the other's length).
myX=[1,2,3]
myY=[3,2,1]
dot2=Dotproduct(myX,myY)
print(f"Dotproduct with my example: {dot2}")