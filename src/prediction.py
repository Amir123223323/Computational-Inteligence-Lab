import numpy as np
from src.activation_function import sigmoid

def predict(w,  X, bias = 0):
    X = np.asarray(X, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    z = np.dot(X, w) + bias
    output = sigmoid(z)
    prediction = (output >= 0.5).astype(int)
    return prediction.reshape(-1)