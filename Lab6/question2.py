# Data normalization - scale the values between 0 and 1. Implement code from scratch.
# test example.
import sys
import pandas as pd
X=[23,34,24,24,40,10,13,23,67]
nX=[((x-min(X))/(max(X)-min(X))) for x in X]
# print(nX)

# we will try to normalize every data point in simulated dataset
data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
Xdat= data.drop(columns=["disease_score","disease_score_fluct"]).values
print(Xdat)


def normalize(X):
    norm=[]
    for i in range(len(X)):
        set=[]
        for j in range(len(X[i])):
            normie=((X[i][j]-min(X[i]))/(max(X[i])-min(X[i])))
            set.append(normie)
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

trns=transpose(Xdat) # to change the dimension from mXn to nXm
trns_norm=normalize(trns) # now easy to find the min and max
norm=transpose(trns_norm) # flip it back
print(pd.DataFrame(norm))

