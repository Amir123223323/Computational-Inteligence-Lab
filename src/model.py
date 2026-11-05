from pre_processing import  _pre_processing
from perceptron_algorithm import binary_classifier
from prediction import predict
from sklearn.model_selection import train_test_split

def train_model(x,y,epochs = 1000,learning_rate = 0.001,seed = 42,bias = True):
    x, y = _pre_processing(x, y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
    weights , loss_function , bias_ = binary_classifier(x_train,
                                                       y_train,
                                                       epochs=epochs,
                                                       learning_rate=learning_rate,
                                                       seed_=seed)
    y_pred = predict(weights,x_test,bias= bias_) if bias else predict(weights,x_test)
    return y_pred , y_test , loss_function , weights ,bias