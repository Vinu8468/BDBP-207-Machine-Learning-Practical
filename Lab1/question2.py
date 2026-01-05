# Implement y = 2x1 + 3 and plot x1, y [start=-100, stop=100, num=100]
start=-100
stop=100
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
    yval=2*n+3
    y.append(yval)

plt.plot(x1,y)
plt.xlabel("X1")
plt.ylabel("Y")
plt.title("y = 2*x1 + 3")
plt.show()
