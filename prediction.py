"""Prediction on the data set"""
import pandas as pd

from classifiers.naive_bayes import NaiveBayes
from classifiers.util import train_test_split, confusion_matrix, accuracy_score

# reading the data set
df = pd.read_csv("./dataset/processed.csv")
X = df.drop("Outcome", axis=1).values
y = df["Outcome"].values

# making sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20)

# training the model
nb = NaiveBayes(X_train, y_train)
nb.fit(X_train, y_train)

# getting the predictions
y_predictions = nb.predict(X_test)
print(f"The accuracy score :: {accuracy_score(y_predictions, y_test) * 100} %")

# confusion matrix
print("Confusion Matrix ::")
print(f"{confusion_matrix(y_test, y_predictions)}")
