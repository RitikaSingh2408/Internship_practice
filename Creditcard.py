import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

df = pd.read_csv('clean_creditcard.csv')
# === check frount vs non-Fraud  count===
class_count = df["Class"].value_counts().sort_index()
print(class_count)
# plt.figure(figsize=(6,5))
# plt.scatter(class_count.index,
#              class_count.values,
#              s=250,
#              color=["blue" , "red"])
# plt.plot(class_count.index, class_count.values)
# plt.xticks([0,1],["Non-Fraud", "Fraud"])
# plt.xlabel("transaction Class")
# plt.ylabel("Number of transactions")
# plt.title("Class distribution (scatter plot)")
# plt.grid(True)
# plt.show()

# ==== Random Under Sampling ===
# x= df.drop("Class",axis=1)
# y=df["Class"]
# rus = RandomUnderSampler(random_state=42)
# x_under, y_under = rus.fit_resample(x,y)
# print(y_under.value_counts())

# ===== scatter plot after over sampling =====
# x_over = pd.DataFrame(x_over, columns=df.drop("Class",axis=1).columns)
# y_over = pd.Series(y_over, name="Class")
# over_df = pd.concat([x_over, y_over], axis=1)

# plt.figure(figsize=(10,6))

# plt.scatter(
#     over_df["Time"],
#     over_df["Amount"],
#     c=over_df["Class"],
#     alpha=0.6
# )

# plt.xlabel("Time")
# plt.ylabel("Amount")
# plt.title("Scatter Plot After Random Over Sampling")

# plt.colorbar(label="Class (0=Non-Fraud, 1=Fraud)")

# plt.show()

# comparison = pd.DataFrame({
#     "Actual": y_test.values,
#     "Predicted": y_pred
# })

# print(comparison.head(20))

# correct = (y_test.values == y_pred).sum()
# incorrect = (y_test.values != y_pred).sum()

# print("Correct Predictions:", correct)
# print("Incorrect Predictions:", incorrect)
total_transactions = len(df)
# duplicate_records = df.duplicated().sum()
# print("Total duplicate :", duplicate_records)
# df = df.drop_duplicates()

# print("Remaining Duplicates:", df.duplicated().sum())
# print(df.shape)
# print(df.describe())
# missing_values = df.isnull().sum()
# print("Missing values:", missing_values)
# print("Total Missing Values:", df.isnull().sum().sum())

# ==== Outlier Detection ====
# Q1 = df["Amount"].quantile(0.25)
# Q3 = df["Amount"].quantile(0.75)

# IQR = Q3 - Q1

# lower = Q1 - 1.5 * IQR
# upper = Q3 + 1.5 * IQR

# outliers = df[(df["Amount"] < lower) | (df["Amount"] > upper)]

# print("Total Outliers:", len(outliers))

# ==== Box plot ====

# plt.figure(figsize=(8,5))
# sns.boxplot(x=df["Amount"])
# plt.title("Box plot of transaction amount")
# plt.show()

# ==== correlation matrix and heatmap ====

# correlation = df.corr()
# print(correlation)
# plt.figure(figsize=(18,12))
# sns.heatmap(correlation,
#             cmap="coolwarm",
#             center=0)
# plt.title("Correlation Heatmap")
# plt.show()

# ===== highly Correlated Features =====
# fraud_corr = correlation["Class"].sort_values(ascending=False)
# print(fraud_corr)

# top_corr = fraud_corr.head(10)
# print("top correated:",top_corr)
# print("total transactions:", total_transactions)

# fraud_transactions = df[df["Class"] == 1].shape[0]
# print("Total fraud transactions:", fraud_transactions)

# genuine_transactions = df[df["Class"] == 0].shape[0]
# print("Total Genuine Transactions:", genuine_transactions)

# fraud_percentage = (fraud_transactions / total_transactions) * 100
# print("fraud percentage:", round(fraud_percentage,2), "%")

# print("Maximum Transaction", df["Amount"].max())
# print("Minimum Transaction", df["Amount"].min())
# print("Average Transaction", df["Amount"].mean())
# print("Median Transaction", df["Amount"].median())

# df["Hour"] = (df["Time"] // 3600) % 24
# print(df[["Time", "Hour"]].head())

# fraud_by_hour = df[df["Class"] ==1].groupby("Hour").size()
# print(fraud_by_hour)

# peak_hour = fraud_by_hour.idxmax()
# print("Peak Fraud Hour:", peak_hour)

# fraud_by_hour.plot(kind="bar", figsize=(10,5))
# plt.title("Fraud Transactons by Hour")
# plt.xlabel("Hour")
# plt.ylabel("Fraud Count")
# plt.show()

# ==== count plot ====
# plt.figure(figsize=(6,4))
# sns.countplot(x="Class", data=df)
# plt.title("Fraud vs Genuine Transactions")
# plt.show()

# === Histogram ===

# plt.figure(figsize=(8,5))
# plt.hist(df["Amount"], bins=50)
# plt.title("Transaction Amount Distribution")
# plt.xlabel("Amount")
# plt.ylabel("Frequency")
# plt.show()

# === box plot ===

# plt.figure(figsize=(7,5))
# sns.boxplot(x="Class", y="Amount", data=df)
# plt.title("Amount by Transaction Type")
# plt.show()

# === pie chart ===
# counts = df["Class"].value_counts()
# plt.figure(figsize=(6,6))

# plt.pie(
#     counts,
#     labels=["Genuine", "Fraud"],
#     autopct = "%1.2f%%",
#     startangle=90
# )
# plt.title("transaction Distribution")
# plt.show()

# ==== Distribution plot ====

# plt.figure(figsize=(8,5))
# sns.histplot(df["Amount"], kde=True)
# plt.title("Amount Distribution")
# plt.show()

# ==== standard scalar ====

standard_scaler = StandardScaler()
df_standard = df.copy()
df_standard["Amount"] = standard_scaler.fit_transform(df_standard[["Amount"]])
print(df_standard["Amount"].head(10))

# ==== minmax Scaler ====

minmax_scaler = MinMaxScaler()
df_minmax = df.copy()
df_minmax["Amount"]= minmax_scaler.fit_transform(df_minmax[["Amount"]])
print(df_minmax["Amount"].head(10))

# ==== Compare both methods ====

print("Original Amount")
print(df["Amount"].head())

print("\nStandardScaler")
print(df_standard["Amount"].head())

print("\nMinMaxScaler")
print(df_minmax["Amount"].head())

# ==== check any categorial columns ====
# print(df.select_dtypes(include="object").columns)

# ===== class imbalance analysis 1.count fraud vs non fraud =====
class_count = df["Class"].value_counts()
print(class_count)

# ==== plot class distribution ====
# plt.figure(figsize=(6,4))
# sns.countplot(x="Class", data=df)
# plt.title("Fraud vs non-Fraud Transactions")
# plt.xlabel("Class")
# plt.ylabel("Count")
# plt.show()
 
#  === fraud percentage ====

# fraud_percentage = (df["Class"].sum() / len(df)) * 100
# print("fraud percentage:", round(fraud_percentage,2), "%")

# ->output 0.17% fraud transaction

# ===define features and target ===

x = df.drop("Class", axis=1)
print(x.head())
y = df["Class"]

scaler = StandardScaler()
x = scaler.fit_transform(x)

# === split the dataset ===

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

# === train the logostic regression mpodel ===

model = LogisticRegression(max_iter=5000)
model.fit(x_train, y_train)

# ==== make predictions ====

y_pred = model.predict(x_test)
print(y_pred)
print(classification_report(y_test,y_pred))

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

cm = (confusion_matrix(y_test, y_pred))
print(cm)
plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Non-Fraud','Fraud'],
    yticklabels=['Non-Fraud','Fraud']
)
plt.title("Confusion matrix")
plt.xlabel("preditcted label")
plt.ylabel("Actual label")
plt.show()
print(classification_report(y_test, y_pred))

'''           precision    recall  f1-score   support

           0       1.00      1.00      1.00     56656
           1       0.89      0.56      0.68        90

    accuracy                           1.00     56746
   macro avg       0.95      0.78      0.84     56746
weighted avg       1.00      1.00      1.00     56746

Accuracy: 0.9991893701758714
[[56650     6]
 [   40    50]] '''

# ==== perform sampling ====
x=df.drop("Class",axis=1)
y= df["Class"]
scaler = StandardScaler()
x=scaler.fit_transform(x)

# ===== Random over Sampling =====

ros = RandomOverSampler(random_state=42)
x_over, y_over = ros.fit_resample(x,y)
print(y_over.value_counts())

# ==== Train Logistic Regression Again ====
x_train, x_test, y_train, y_test = train_test_split(
    x_over,
    y_over,
    test_size=0.2,
    random_state=42
)
model = LogisticRegression(max_iter=5000,class_weight="balanced",random_state=42)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(classification_report(y_test,y_pred))
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

cm = (confusion_matrix(y_test, y_pred))
print(cm)
correct = (y_test.values == y_pred).sum()
incorrect = (y_test.values != y_pred).sum()

print("Correct Predictions:", correct)
print("Incorrect Predictions:", incorrect)

'''            precision    recall  f1-score   support

           0       0.93      0.98      0.95     56463
           1       0.98      0.92      0.95     56839

    accuracy                           0.95    113302
   macro avg       0.95      0.95      0.95    113302
weighted avg       0.95      0.95      0.95    113302

Accuracy: 0.949153589521809
[[55170  1293]
 [ 4468 52371]]
 Correct Predictions: 107541
Incorrect Predictions: 5761'''

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# ==== decision tree ====
dt_model = DecisionTreeClassifier(
    max_depth=10,
    class_weight="balanced",
    random_state=42
)
dt_model.fit(x_train, y_train)
dt_pred =dt_model.predict(x_test)
print("Decision Tree")
print("Accuracy:", accuracy_score(y_test, dt_pred))
print("Precision:", precision_score(y_test, dt_pred))
print("Recall:", recall_score(y_test, dt_pred))
print("F1 Score:", f1_score(y_test, dt_pred))

print("\nClassification Report:")
print(classification_report(y_test, dt_pred))

'''Accuracy: 0.9967696951510123
Precision: 0.9936019578708155
Recall: 1.0
F1 Score: 0.9967907123566343

Classification Report:
              precision    recall  f1-score   support

           0       1.00      0.99      1.00     56463
           1       0.99      1.00      1.00     56839

    accuracy                           1.00    113302
   macro avg       1.00      1.00      1.00    113302
weighted avg       1.00      1.00      1.00    113302 '''

# ===== Random Forest classifier ====
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
rf_model.fit(x_train, y_train)
rf_pred = rf_model.predict(x_test)

print("Random forest")
print("Accuracy:",accuracy_score(y_test,rf_pred))
print("precision:", precision_score(y_test, rf_pred))
print("Recall:", recall_score(y_test,rf_pred))
print("F1 score:", f1_score(y_test,rf_pred))
print("classification report:")
print(classification_report(y_test, rf_pred))

'''Accuracy: 0.9999735220914018
precision: 0.9999472221244854
Recall: 1.0
F1 score: 0.9999736103658483
classification report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56463
           1       1.00      1.00      1.00     56839

    accuracy                           1.00    113302
   macro avg       1.00      1.00      1.00    113302
weighted avg       1.00      1.00      1.00    113302
'''

# # ==== KNN ====
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train_scaled,y_train)
knn_pred = knn_model.predict(x_test_scaled)

print("KNN")
print("Accuracy:", accuracy_score(y_test, knn_pred))
print(classification_report(y_test,knn_pred))

'''Accuracy: 0.9996822650968209
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56463
           1       1.00      1.00      1.00     56839

    accuracy                           1.00    113302
   macro avg       1.00      1.00      1.00    113302
weighted avg       1.00      1.00      1.00    113302 
'''

# # ==== compare all algorithm ====
results = pd.DataFrame({
    "Algorithm": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "KNN"
    ],
    "Accuracy": [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, dt_pred),
        accuracy_score(y_test, rf_pred),
        accuracy_score(y_test, knn_pred)
    ],
    "Precision": [
        precision_score(y_test, y_pred),
        precision_score(y_test, dt_pred),
        precision_score(y_test, rf_pred),
        precision_score(y_test, knn_pred)
    ],
    "Recall": [
        recall_score(y_test, y_pred),
        recall_score(y_test, dt_pred),
        recall_score(y_test, rf_pred),
        recall_score(y_test, knn_pred)
    ],
    "F1 Score": [
        f1_score(y_test, y_pred),
        f1_score(y_test, dt_pred),
        f1_score(y_test, rf_pred),
        f1_score(y_test, knn_pred)
    ]
})

print(results)


# ==== comparison graph ====
results.set_index("Algorithm")[["Precision", "Recall", "F1 Score"]].plot(
    kind="bar",
    figsize=(10,6)
)

plt.title("Algorithm Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)

plt.xticks(rotation=0)
plt.legend()

plt.show()
  
'''  ==== algorithm comparison table Output ====
#              Algorithm  Accuracy  Precision    Recall  F1 Score
# 0  Logistic Regression  0.949154   0.975906  0.921392  0.947866
# 1        Decision Tree  0.996770   0.993602  1.000000  0.996791
# 2        Random Forest  0.999974   0.999947  1.000000  0.999974
# 3                  KNN  0.999682   0.999367  1.000000  0.999683
'''

print("Decision Tree Confusion Matrix:")
print(confusion_matrix(y_test, dt_pred))

'''[[56097   366]
   [    0 56839]]
 '''
# # ==== confusion matrix by Random Forest ====
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

'''[[56460     3]
   [    0 56839]]  
   '''

# # ==== confusion matrix by KNN ====
print("KNN Confusion Matrix:")
print(confusion_matrix(y_test, knn_pred))

'''[[56427    36]
   [    0 56839]]
   '''

print(df.head())

# === confusion metrix graphical representation ====

models = {
    "Logistic Regression": cm,
    "Decision Tree": dt_pred,
    "Random Forest": rf_pred,
    "KNN": knn_pred
}

for name, prediction in models.items():

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        prediction
    )

    plt.title(name)
    plt.show()