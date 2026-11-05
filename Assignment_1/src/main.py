from plot_metrics import plot_confusion_matrix,plot_loss,plot_weights
from model import train_model
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)


def main():
    from src.load_data_set import X,y
    y_pred , y_test , loss_function , weights , bias = train_model(X,y,bias= True)
    y_test = np.asarray(y_test, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    print(y_pred )
    print(y_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(weights)
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Not Survived", "Survived"]
    ))
    plot_weights(weights)
    plot_confusion_matrix(y_test, y_pred)
    plot_loss(loss_function)


if __name__ == '__main__':
    main()
