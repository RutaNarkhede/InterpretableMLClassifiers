import numpy as np
import mcless

def predict_y(X,W):
    A = Information_matrix(X)
    B_pred = A @ W
    print(B_pred.shape)
    print(B_pred[0])
    N = len(B_pred)
    y_pred = np.zeros(N)
    for i in range(N):
        c = np.argmax(B_pred[i])
        y_pred[i] = c
    return y_pred