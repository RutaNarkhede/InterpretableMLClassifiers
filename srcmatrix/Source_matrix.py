import numpy as np

def Source_matrix(y):
    N = len(y)
    allclasses = np.unique(y)
    B = np.zeros((N,len(allclasses)))
    for i in range(N):
        class_value = int(y[i])
        B[i][class_value] = 1
    return B