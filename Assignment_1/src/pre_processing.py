import numpy as np



def  _pre_processing(X,y):
    X[:, 1] = np.where(X[:, 1] == "female", 0, 1)
    print("Sex evaluation is complete...")
    X[:,0] = X[:,0] / max(X[:,0])
    print("Pclass normalization is complete...")
    X[:, 2] = X[:, 2] / max(X[:, 2])
    print("SibSp normalization is complete...")
    X[:, 3] = X[:, 3] / max(X[:, 3])
    print("Parch normalization is complete...")
    X[:, 4] = X[:, 4] / max(X[:, 4])
    print("Fare normalization is complete...")
    y[:,] = y[:] / max(y[:])
    print("Survived normalization is complete...")
    return X,y



