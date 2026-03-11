# Implement information gain measures. The function should accept
# data points for parents, data points for both children and return an
# information gain value.

import numpy as np
from collections import Counter
def entropy(labels):
    counts = Counter(labels)
    total = len(labels)
    entropy = 0
    for count in counts.values():
        p = count/total
        entropy = entropy - p*np.log2(p)
    return entropy

def information_gain(parent_label, left_label, right_label):
    parent_entropy = entropy(parent_label)
    n = len(parent_label)
    n_left = len(left_label)
    n_right = len(right_label)
    child_entropy = (n_left/n)*entropy(left_label)+ \
                    (n_right/n)*entropy(right_label)
    ig = parent_entropy - child_entropy
    return ig

parent = ["A","A","B","B","A","B"]
left= ["A",'A',"A"]
right = ["B",'B',"B"]

print(information_gain(parent,left,right))