import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Data Collection
df = pd.read_csv(r"C:\Zaman_btech\Zaman_btech\Customer Segmentation Analysis (level 1)\ifood_df.csv")
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
print(df.isnull().sum()) # No Missing Values

print("\nDuplicate Records: ")
print(df.duplicated().sum()) # 184
df.drop_duplicates(inplace=True)

# Descriptive Statistics
print("\nAverage Income: ")
print(df['Income'].mean())
print("\nAverage Spending: ")
print(df['MntTotal'].mean())
print("\nAverage Age: ")
print(df['Age'].mean())
print("\nPurchase Statistics: ")
print(df[['NumDealsPurchases','NumWebPurchases','NumStorePurchases','NumCatalogPurchases']].describe())

# Visualization (Before Clustering)

# Income Distribution
plt.figure(figsize=(8,5))
plt.hist(df['Income'], bins=20)
plt.title("Income Distribution")
plt.xlabel("Income")
plt.ylabel("Customers")
plt.show()

# Age Distribution
plt.figure(figsize=(8,5))
plt.hist(df['Age'], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Customers")
plt.show()

# Total Spending Distribution
plt.figure(figsize=(8,5))
plt.hist(df['MntTotal'], bins=20)
plt.title("Customer Spending Distribution")
plt.xlabel("Total Spending")
plt.ylabel("Customers")
plt.show()

# Purchases by Channel
purchase_data = {
    'Web': df['NumWebPurchases'].sum(),
    'Store': df['NumStorePurchases'].sum(),
    'Catalog': df['NumCatalogPurchases'].sum(),
}
plt.figure(figsize=(7,5))
plt.bar(purchase_data.keys(),purchase_data.values())
plt.title("Purchases by Channel")
plt.xlabel("Channel")
plt.ylabel("Total Purchases")
plt.show()

# Correlation Matrix
plt.figure(figsize=(10,6))
corr = df[['Income','Age','MntTotal','NumWebPurchases','NumStorePurchases','NumCatalogPurchases']].corr()
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr.columns)),corr.columns,rotation=90)
plt.yticks(range(len(corr.columns)),corr.columns)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# Customer Segmentation (K-Means)
x = df[['Income','MntTotal']]
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,random_state=42,n_init=10)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11),wcss,marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

kmeans=KMeans(n_clusters=4,random_state=42,n_init=10)
df['Cluster'] = kmeans.fit_predict(x_scaled)

# Segment Visualization
plt.figure(figsize=(10,6))
plt.scatter(df['Income'],df['MntTotal'],c = df['Cluster'])
plt.title("Customer Segmentation using K-Means")
plt.xlabel("Income")
plt.ylabel("Total Spending")
plt.show()

cluster_spending = df.groupby('Cluster')['MntTotal'].mean()
plt.figure(figsize=(7,5))
plt.bar(cluster_spending.index.astype(str),cluster_spending.values)
plt.title("Average Spending by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Spending")
plt.show()

cluster_summary = df.groupby('Cluster')[['Income','MntTotal','Age']].mean()
print(cluster_summary)

df.to_csv("Customer_Segmentation_Result.csv",index=False)
print("Project Completed Successfully!")