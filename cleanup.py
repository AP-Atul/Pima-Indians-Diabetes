"""Cleaning up the data set"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# reading the data set
df = pd.read_csv("./dataset/diabetes.csv")

# features in the data set
# OP: 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
#     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
print(df.columns)

# data rows
# OP : 768
print(len(df))

# exception feature
dff = df.drop(["Outcome", "Pregnancies"], axis=1)

# replacing the 0 -> na
dff = dff.replace(0, np.NAN)
print(f"Total number of null data :: {dff.isnull().sum()}")

# bringing back data
df = pd.concat([dff, df["Pregnancies"], df["Outcome"]], axis=1)

# replacing na with the median of their outcome type
for i in df.columns:
    df[i] = df[i].fillna(df.groupby("Outcome")[i].transform("median"))
print(f"Total number of data :: {len(df)}")

# changing the data type of Outcome
df['Outcome'] = df.Outcome.astype("category")
print(f"Data types :: {df.dtypes}")

# plotting the data frame
df.hist()
plt.show()

# saving the processed csv
df.to_csv("./dataset/processed.csv", index=False)
