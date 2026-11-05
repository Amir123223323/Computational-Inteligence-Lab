import pandas as pd
import numpy as np
csv_path = r"src\titanic.csv"
Frame = pd.read_csv(csv_path)
print("Titanic .csv file has successfully loaded...")
data = Frame.values
features = {"Pclass":2,"Sex":4,"SibSp":6,"Parch":7,"Fare":9}
outputs = {"Survived":1}
X = np.column_stack((data[:,features["Pclass"]]
                         ,data[:,features["Sex"]]
                     ,data[:,features["SibSp"]]
                     ,data[:,features["Parch"]]
                     ,data[:,features["Fare"]]))
print("Features matrix has successfully loaded...\n")
print(f"Features: {features.keys()}\n")
y = data[:,outputs["Survived"]]
print("Output vector has successfully loaded...")
print(f"Outputs: {outputs.keys()}\n")


