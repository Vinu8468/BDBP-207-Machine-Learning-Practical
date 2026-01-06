# Implement Gaussian PDF - mean = 0, sigma = 15 in the range[start=-100, stop=100, num=100]
import math
import matplotlib.pyplot as plt

mean = 0
sigma = 15
start = -100
stop = 100
num = 100

step = (stop - start) / (num - 1)
x = []
s = start
for n in range(num):
    x.append(s)
    s+= step


y = []
for val in x:
    pdf = (1 / (sigma * math.sqrt(2 * math.pi))) *  math.exp(-((val - mean) ** 2) / (2 * sigma ** 2))
    y.append(pdf)

plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("Gaussian PDF")
plt.title("Gaussian PDF (mean=0, sigma=15)")
plt.show()
