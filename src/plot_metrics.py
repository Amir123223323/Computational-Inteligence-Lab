import  matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plot_weights(w):
    Feature = ["Pclass","Sex","SibSp","Parch","Fare"]
    weights = w.flatten()
    plt.bar(Feature, weights)
    plt.xlabel("Feature")
    plt.ylabel("Weight")
    plt.title("Model Weights")
    plt.xticks(range(len(weights)))
    plt.grid(axis="y", alpha=0.3)
    plt.show()



def plot_confusion_matrix(y_test,y_pred):
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["Not Survived", "Survived"],
        cmap="Blues"
    )
    plt.show()


def plot_loss(loss):
    plt.plot(loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid()
    plt.show()
