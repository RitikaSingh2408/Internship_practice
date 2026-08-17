import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
df = pd.read_csv('creditcard.csv')
print(df.head(10))
print(df.shape)
print(df.info())
print(df.columns)
print(df.describe())
print(df.dtypes)
# ==== missing value ====
print(df.isnull().sum())
print(df.duplicated().sum()) 
# ==== remove duplicates ====
df = df.drop_duplicates()
print(df.duplicated().sum())
# ===== check class distribution ====
print(df["Class"].value_counts())

# ==== find percentage ====
print(df["Class"].value_counts(normalize=True)*100)

# ==== genuin vs fraud transaction ====
sns.countplot(x="Class", data=df)

plt.title("Fraud vs Genuine Transactions")
plt.show()

# ==== transation amount distribution ===
plt.figure(figsize=(8,5))

sns.histplot(df["Amount"], bins=50)

plt.title("Transaction Amount Distribution")

plt.show()

# === fraud amount distribution ===
sns.boxplot(x="Class", y="Amount", data=df)

plt.title("Fraud Transaction Amount")

plt.show()
(df.to_csv("clean_creditcard.csv", index=False))
print(df.head(10))
print("dataset succesfull saved")