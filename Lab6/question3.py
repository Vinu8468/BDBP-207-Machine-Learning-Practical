# Data standardization - scale the values such that mean of new dist = 0 and sd = 1. Implement code from scratch.
import numpy as np
import pandas as pd
X=[23,34,24,24,40,10,13,23,67]
meanx=np.mean(X)
stdx = np.std(X)
sdx=[(x-meanx)/stdx for x in X]
sum=np.mean(sdx)
sd=np.std(sdx)
print(sum)
print(sd)


# we will try to standardize every data point in simulated dataset
data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
Xdat= data.drop(columns=["disease_score","disease_score_fluct"]).values
print(Xdat)


def standardize(X):
    norm=[]
    for i in range(len(X)):
        set=[]
        for j in range(len(X[i])):
            stanval=((X[i][j]-np.mean(X[i]))/(np.std(X[i])))
            set.append(stanval)
        norm.append(set)
    return norm


def transpose(X):
    transpose=[]
    for i in range(len(X[0])):
        set=[]
        for j in range(len(X)):
            set.append(X[j][i])
        transpose.append(set)
    return transpose

tnx=transpose(Xdat)
stn_tnx=standardize(tnx)
standardized= transpose(stn_tnx)
print(pd.DataFrame(standardized))