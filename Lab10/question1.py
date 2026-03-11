# Implement entropy measure using Python. The function should accept a
# set of data points and their class labels and return
# the entropy value.

from collections import Counter
import numpy as np

def entropy(labels):
    counts = Counter(labels)
    total = len(labels)
    entropy = 0
    for count in counts.values():
        p = count/total
        entropy = entropy - p*np.log2(p)
    return entropy

labels = ["A","B","A","A","A","B","C"]
print(entropy(labels))

