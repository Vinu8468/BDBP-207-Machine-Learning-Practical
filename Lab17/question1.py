#Implement a feature mapping function called Transform()
# converting 2D to 3D
# phi(x1,x2)=((x1)2,sqrt(2x1.x2),(x2)2)

# starting of with loading the dataset which is csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv("sampledata.csv")
print(data.head())

blue = data[data["Label"] == "Blue"]
red = data[data['Label'] == 'Red']

# now we plot this 2D tingy
plt.scatter(blue["x1"], blue["x2"], label="Blue",color = "blue")
plt.scatter(red["x1"], red["x2"], label="Red",color = "red")

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("2D data plot")
plt.legend()

# plt.show()

def transform(x1,x2):
    return np.array([x1**2,np.sqrt(2)*x1*x2,x2**2])

transformed = data.apply(lambda row: transform(row["x1"],row["x2"]), axis=1)
# so this will send the data points to the function which will return 3 features for 3 D analysis

transformed_df = pd.DataFrame(transformed.tolist(),columns=["z1",'z2',"z3"])
# this is just making a proper dataframe with heading

transformed_df["Label"] = data["Label"]

print(transformed_df.head())

import matplotlib.pyplot as plt

blue = transformed_df[transformed_df["Label"] == "Blue"]
red = transformed_df[transformed_df["Label"] == "Red"]
fig = plt.figure()
ax= fig.add_subplot(111, projection="3d")

# plot points
ax.scatter(blue["z1"],blue["z2"],blue["z3"],label="Blue")
ax.scatter(red["z1"],red["z2"],red["z3"],label="Red")

# labels
ax.set_xlabel("z1= x1^2")
ax.set_ylabel("z2= sqrt(2)x1x2")
ax.set_zlabel("z3= x2^2")

ax.set_title("3D data plot")
ax.legend()

plt.show()


















