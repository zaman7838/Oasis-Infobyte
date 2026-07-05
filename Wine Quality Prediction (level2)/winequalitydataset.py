import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Data Collection
df = pd.read_csv(r"C:\Zaman_btech\Zaman_btech\Wine Quality Prediction (level2)\WineQT.csv")
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
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())
df = df.drop_duplicates()
if 'Id' in df.columns:
    df.drop('Id', axis=1, inplace=True)
print("\nDataset Shape After Cleaning:")
print(df.shape)

print("\nStatistical Summary: ")
print(df.describe())
print("\nWine Quality Distribution: ")
print(df['quality'].value_counts().sort_index())
print("\nCorrelation Matrix: ")
print(df.corr())
# Wine Quality Distribution
plt.figure(figsize=(8,5))
sns.countplot(x='quality', data=df)
plt.title("Distribution of Wine Quality")
plt.xlabel("Wine Quality")
plt.ylabel("Count")
plt.show()
# Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
# Alcohol Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['alcohol'], bins=20, kde=True)
plt.title("Alcohol Distribution")
plt.xlabel("Alcohol")
plt.ylabel("Frequency")
plt.show()
# Density Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['density'], bins=20, kde=True)
plt.title("Density Distribution")
plt.xlabel("Density")
plt.ylabel("Frequency")
plt.show()
# Boxplot
plt.figure(figsize=(8,5))
sns.boxplot(x=df['quality'])
plt.title("Boxplot of Wine Quality")
plt.show()
# Histograms
df.hist(figsize=(15,12), bins=20)
plt.suptitle("Distribution of All Features")
plt.tight_layout()
plt.show()

# Feature Selection & Train-Test Split
X = df.drop('quality', axis=1)
y = df['quality']
print("\nFeature Matrix Shape:", X.shape)
print("Target Vector Shape:", y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("\nTraining Data Shape:")
print(X_train.shape)
print("\nTesting Data Shape:")
print(X_test.shape)
print("\nTraining Target Shape:")
print(y_train.shape)
print("\nTesting Target Shape:")
print(y_test.shape)
# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\nFeature Scaling Completed Successfully!")
print("Scaled Training Data Shape:", X_train_scaled.shape)
print("Scaled Testing Data Shape:", X_test_scaled.shape)

# Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("\nRandom Forest Classifier")
print("Accuracy:", round(rf_accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, rf_predictions))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))

# SGD Classifier
sgd_model = SGDClassifier(
    random_state=42,
    max_iter=1000,
    tol=1e-3
)
sgd_model.fit(X_train_scaled, y_train)
sgd_predictions = sgd_model.predict(X_test_scaled)
sgd_accuracy = accuracy_score(y_test, sgd_predictions)
print("\nSGD Classifier")
print("Accuracy:", round(sgd_accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, sgd_predictions))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, sgd_predictions))

# Support Vector Classifier
svc_model = SVC(
    kernel='rbf',
    random_state=42
)
svc_model.fit(X_train_scaled, y_train)
svc_predictions = svc_model.predict(X_test_scaled)
svc_accuracy = accuracy_score(y_test, svc_predictions)
print("\nSupport Vector Classifier")
print("Accuracy:", round(svc_accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, svc_predictions))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, svc_predictions))

# Model Comparison
comparison = pd.DataFrame({
    "Model": [
        "Random Forest",
        "SGD Classifier",
        "Support Vector Classifier"
    ],
    "Accuracy (%)": [
        round(rf_accuracy * 100, 2),
        round(sgd_accuracy * 100, 2),
        round(svc_accuracy * 100, 2)
    ]
})
print("\nModel Comparison")
print(comparison)
best_model = comparison.loc[comparison["Accuracy (%)"].idxmax()]
print("\nBest Model:")
print(best_model)

# Accuracy Comparison Graph
plt.figure(figsize=(8,5))
plt.bar(comparison["Model"], comparison["Accuracy (%)"])
plt.title("Model Accuracy Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy (%)")
plt.xticks(rotation=10)
plt.tight_layout()
plt.show()

# Feature Importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})
importance = importance.sort_values(
    by="Importance",
    ascending=False
)
print("\nFeature Importance:")
print(importance)
plt.figure(figsize=(10,6))
sns.barplot(
    data=importance,
    x="Importance",
    y="Feature"
)
plt.title("Feature Importance (Random Forest)")
plt.show()