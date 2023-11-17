import numpy as np


a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


idx = np.array([[0, 2], [0, 2]])


# print(a)
# print(idx)
a[idx[0, :], idx[1, :]] += 1
# print(a[idx[0, :], idx[1, :]])
print(a)