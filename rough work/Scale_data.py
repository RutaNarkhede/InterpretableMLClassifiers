
import numpy as np

def scale_factor(X):
    X = X.transpose()
    scale = np.zeros((len(X)))
    for i in range(len(X)):
        row = X[i]
        scale[i] = np.amax(np.abs(row))
    return scale

def scale_data(X,scale):
    X= X.transpose()
    scaled_X = np.zeros((X.shape))
    
    for i in range(len(X)):
        row = X[i]
        scaled_X[i] = X[i]/scale[i]
    
    return scaled_X.transpose()





