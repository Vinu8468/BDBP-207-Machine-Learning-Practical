# Implement y = 2x12 + 3x1 + 4 and plot x1, y in the range [start=--10, stop=10, num=100]
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
    yval=2*(n**2)+(3*n)+4
    y.append(yval)

plt.plot(x1,y)
plt.xlabel("X1")
plt.ylabel("Y")
plt.title("y = 2*(n**2)+(3*n)+4")
plt.show()
