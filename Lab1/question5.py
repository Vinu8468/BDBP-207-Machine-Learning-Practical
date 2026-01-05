# Implement y = x1^2, plot x1, y in the range [start=--10, stop=10, num=100]. 
# Compute the value of derivatives at these points, x1 = -5, -3, 0, 3, 5. 
#  What is the value of x1 at which the function value (y) is zero.
#  What do you infer from this?
start=-10
stop=10
num=100
step=(stop-start)/(num-1)

import matplotlib.pyplot as plt
x1=[]
s=start
for n in range(num):
    x1.append(s)
    s+=step
y=[]
for n in x1:
    yval=n**2
    y.append(yval)

plt.plot(x1,y)
plt.xlabel("X1")
plt.ylabel("Y")
plt.title("y = n**2")
plt.show()
