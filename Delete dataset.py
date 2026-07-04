import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# Data Collection
df = pd.read_csv(r"C:\Zaman_btech\Zaman_btech\Fraud Detection (level 2)\creditcard.csv")
print("\nFirst 5 rows: ")
print(df.head())

print("\nLast 5 rows: ")
print(df.tail())

print("\nDataset Shape: ")
print(df.shape)

print("\nColumn Names: ")
print(df.columns)

print("\nDataset Information: ")
print(df.info())

# Data Cleaning
print("\nMissing Values: ")
print(df.isnull().sum())
duplicates = df.duplicated().sum()
print("Duplicate Rows:", duplicates)
df = df.drop_duplicates()
print("Dataset Shape After Removing Duplicates:")
print(df.shape)
print(df.dtypes)
print(df.describe())
print(df['Class'].value_counts())
fraud = df['Class'].value_counts()[1]
normal = df['Class'].value_counts()[0]
print("Fraud Transactions: ", fraud)
print("Normal Transactions: ", normal)
print("Fraud Percentage:",round((fraud/(fraud+normal))*100,4),"%")

# Visualization
class_counts = df['Class'].value_counts()
plt.figure(figsize=(6,5))
plt.bar(['Normal','Fraud'],class_counts.values)
plt.title("Transaction Class Distribution")
plt.xlabel("Transaction Type")
plt.ylabel("Number Of Transactions")
plt.show()
plt.figure(figsize=(6,6)) #Fraud Percentage
plt.pie(
    class_counts,
    labels=['Normal','Fraud'],
    autopct='%1.2f%%',
    startangle=90
)
plt.title("Fraud VS Normal Transactions")
plt.show()
plt.figure(figsize=(10,5)) #Transaction Amount Distribution
plt.hist(df['Amount'], bins=50)
plt.title("Transaction Amount Distribution")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.show()
plt.figure(figsize=(10,5)) #Transaction Time Distribution
plt.hist(df['Time'], bins=50)
plt.title("Transaction Time Distribution")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.show()
fraud = df[df['Class']==1]
normal = df[df['Class']==0]
plt.figure(figsize=(10,5))
plt.scatter(
    normal['Amount'],
    normal['Time'],
    s=5,
    alpha=0.3,
    label='Normal'
)
plt.scatter(
    fraud['Amount'],
    fraud['Time'],
    s=20,
    label='Fraud'
)
plt.title("Fraud VS Normal Transactions")
plt.xlabel("Transaction Amount")
plt.ylabel("Transaction Time")
plt.legend()
plt.show()

# Feature Engineering
x = df.drop('Class', axis=1)
y = df['Class']
print("Features Shape:", x.shape)
print("Target Shape:", y.shape)
scaler = StandardScaler()
x[['Time','Amount']]=scaler.fit_transform(x[['Time','Amount']])
print("Feature Scaling Completed")
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,test_size=0.20,
    random_state=42,
    stratify=y
)
print("Training Features:",x_train.shape)
print("Testing Features:",x_test.shape)
print("Training Labels:",y_train.shape)
print("Testing Labels:",y_test.shape)
print("Training Set Class Distribution:")
print(y_train.value_counts())
print("\nTesting Set Class Distribution:")
print(y_test.value_counts())

# Logistic Regression Model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(x_train, y_train)
y_pred_lr = lr_model.predict(x_test)
print("===== Logistic Regression =====")
print("Accuracy :", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr))
print("Recall   :", recall_score(y_test, y_pred_lr))
print("F1 Score :", f1_score(y_test, y_pred_lr))
print("ROC-AUC  :", roc_auc_score(y_test, y_pred_lr))
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred_lr))
print("\nClassification Report")
print(classification_report(y_test, y_pred_lr))

# Decision Tree Model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(x_train, y_train)
y_pred_dt = dt_model.predict(x_test)
print("\n===== Decision Tree =====")
print("Accuracy :", accuracy_score(y_test, y_pred_dt))
print("Precision:", precision_score(y_test, y_pred_dt))
print("Recall   :", recall_score(y_test, y_pred_dt))
print("F1 Score :", f1_score(y_test, y_pred_dt))
print("ROC-AUC  :", roc_auc_score(y_test, y_pred_dt))
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred_dt))
print("\nClassification Report")
print(classification_report(y_test, y_pred_dt))

# Random Forest Model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_model.fit(x_train, y_train)
y_pred_rf = rf_model.predict(x_test)
print("\n===== Random Forest =====")
print("Accuracy :", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall   :", recall_score(y_test, y_pred_rf))
print("F1 Score :", f1_score(y_test, y_pred_rf))
print("ROC-AUC  :", roc_auc_score(y_test, y_pred_rf))
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report")
print(classification_report(y_test, y_pred_rf))

# Model Comparison
comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_dt),
        accuracy_score(y_test, y_pred_rf)
    ],
    "Precision": [
        precision_score(y_test, y_pred_lr),
        precision_score(y_test, y_pred_dt),
        precision_score(y_test, y_pred_rf)
    ],
    "Recall": [
        recall_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_dt),
        recall_score(y_test, y_pred_rf)
    ],
    "F1 Score": [
        f1_score(y_test, y_pred_lr),
        f1_score(y_test, y_pred_dt),
        f1_score(y_test, y_pred_rf)
    ],
    "ROC-AUC": [
        roc_auc_score(y_test, y_pred_lr),
        roc_auc_score(y_test, y_pred_dt),
        roc_auc_score(y_test, y_pred_rf)
    ]
})
print(comparison)

# Feature Importance
importance = rf_model.feature_importances_
feature_importance = pd.DataFrame({
    "Feature": x.columns,
    "Importance": importance
})
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)
print(feature_importance.head(10))
top_features = feature_importance.head(10)
plt.figure(figsize=(10,5))
plt.bar(top_features["Feature"],
        top_features["Importance"])
plt.title("Top 10 Important Features")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Conclusion & Key Insights
print("\n========== Project Conclusion ==========")
print(
    "The Credit Card Fraud Detection system was successfully developed using "
    "Machine Learning techniques. After data preprocessing, exploratory data "
    "analysis, feature engineering, and model evaluation, the Random Forest "
    "model achieved the best performance for identifying fraudulent transactions."
)

print("\n========== Key Insights ==========")
print(
    "- The dataset is highly imbalanced, with only 0.17% fraudulent transactions.\n"
    "- Random Forest outperformed Logistic Regression and Decision Tree.\n"
    "- Feature scaling improved model performance.\n"
    "- The project successfully addressed anomaly detection, feature engineering,\n"
    "  machine learning, scalability (concept), and real-time monitoring (concept)."
)




