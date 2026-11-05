import numpy as np
from src.activation_function import sigmoid

def binary_classifier(X, y, epochs=1000,learning_rate=0.001,seed_=42):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    np.random.seed(seed_)
    w = np.random.rand(X.shape[1], 1)
    Loss_function = []
    bias = np.random.rand(1)
    for epoch in range(epochs):
        epoch_cost = 0
        for i in range(len(X)):
            z = np.dot(X[i], w) + bias
            output = sigmoid(z)
            error = output - y[i]
            gradient = error * output * (1 - output)
            w -= learning_rate * gradient * X[i].reshape(-1, 1)
            bias -= learning_rate * gradient
            epoch_cost += error.item()**2 / 2
        Loss_function.append(epoch_cost / len(X))
    return w, Loss_function, bias


