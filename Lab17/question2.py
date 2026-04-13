# Let x1 = [3, 6], x2 = [10, 10].  Use the above “Transform”
# function to transform these vectors to a higher dimension
# and  compute the dot product in a higher dimension. Print the value.
x=[[3,10],[6,10]]
import pandas as pd
import numpy as np
data = pd.DataFrame(x, columns=['x1', 'x2'])
print(data)

def transform(x1,x2):
    return np.array([x1**2,np.sqrt(2)*x1*x2,x2**2])

transformed = data.apply(lambda row: transform(row['x1'],row['x2']), axis=1)
print(transformed)
a = transformed[0]
b = transformed[1]
dot_product = np.dot(a,b)
print(dot_product)
