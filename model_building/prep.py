import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/machine-failure-prediction.csv")
df.drop(columns=["UDI"], inplace=True)

# Encode the categorical 'Type' column
df["Type"] = LabelEncoder().fit_transform(df["Type"])

X = df.drop(columns=["Failure"])
y = df["Failure"]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
