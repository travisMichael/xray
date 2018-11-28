import numpy as np

# a = [[1, 2],[ 3, 4]]
# b = [[6, 7]]
#
#
# c = a + b
#
o = np.ones(5)
z = np.zeros(5)
l = np.vstack((o, z))

m = np.zeros((10,2))
m[0:5, :-1] = 1
# m[:, -1,:]

a = [[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]
# a[0:1]
np.concatenate((a, None))



print("done")
