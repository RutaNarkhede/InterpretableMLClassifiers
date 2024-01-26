import numpy as np

def Information_matrix(X):
    N = len(X)
    A=np.column_stack((np.ones([N,]),X))
    return A