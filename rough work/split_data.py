def split_data(data, splitThreshold=0.3):
    
    rows = data.shape[0]
    cols = data.shape[1]
    
    train_size= int(rows - rows*splitThreshold)
    test_size = int(rows - train_size)
    
    data_train = np.zeros((train_size,cols))
    data_test = np.zeros((test_size,cols))
    
    shuffled_idx = np.empty((rows), dtype='int')
    for i in range(rows):
        shuffled_idx[i] = i
    np.random.permutation(shuffled_idx)
    
    X_train = data[shuffled_idx[0:train_size], 0:cols-1]
    X_test = data[shuffled_idx[train_size:], 0:cols-1]
    y_train = data[shuffled_idx[0:train_size], cols-1]
    y_test = data[shuffled_idx[train_size:], cols-1]
    
    return X_train,X_test,y_train,y_test