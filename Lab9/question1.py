# Write a program to partition a dataset (simulated data for regression)  into two parts,
# based on a feature (BP) and for a threshold, t = 80. Generate additional two partitioned
# datasets based on different threshold values of t = [78, 82].

import pandas as pd

# load the dataset
data = pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')

# function to partition dataset
def partition_dataset(data,feature,threshold):
    left_partition = data[data[feature] <= threshold]
    right_partition = data[data[feature] > threshold]

    print(f"Threshold = {threshold}")
    print(f"Left partition (BP<= threshold):",left_partition.shape)
    print(f"Right partition (BP> threshold):",right_partition.shape)

    return left_partition, right_partition

# threshold values
thresholds =[80,78,82]

for t in thresholds:
    left,right = partition_dataset(data,feature="BP",threshold=t)
# Threshold = 80
# Left partition (BP<= threshold): (9, 7)
# Right partition (BP> threshold): (51, 7)
# Threshold = 78
# Left partition (BP<= threshold): (0, 7)
# Right partition (BP> threshold): (60, 7)
# Threshold = 82
# Left partition (BP<= threshold): (19, 7)
# Right partition (BP> threshold): (41, 7)