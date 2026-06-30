import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data Loading
df = pd.read_csv(r"C:\Zaman_btech\Zaman_btech\EDA on Retail Sales Data (level 1)\menu.csv")

# Data Understanding
print("\nFirst 5 rows: ") # Display First 5 Rows
print(df.head())

print("\nLast 5 rows: ") # Display Last 5 Rows
print(df.tail())

print("\nDataset Information: ")
print(df.info())

print("\nStatistical Summary: ")
print(df.describe())

print("\nNumber of rows and columns: ")
print(df.shape)

print("\nColumn Names: ")
print(df.columns)

# Data Cleaning
print("\nMissing Values: ")
print(df.isnull().sum()) # No Missing Values

print("\nDuplicate Records: ")
print(df.duplicated().sum()) # No Duplicate Records

print("\nData Types: ")
print(df.dtypes)

# Descriptive Statistics
print("\nMean: ")
print(df.mean(numeric_only=True))

print("\nMode: ")
print(df.mode(numeric_only=True))

print("\nMedian: ")
print(df.median(numeric_only=True))

print("\nStandard Deviation: ")
print(df.std(numeric_only=True))

# Product Analysis

top_calories = df.sort_values('Calories', ascending=False).head(10)

plt.figure(figsize=(10,5))
plt.bar(top_calories['Item'], top_calories['Calories'])

plt.title("Top 10 Highest Calorie Items")
plt.xlabel("Menu Items")
plt.ylabel("Calories")

plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

fat_data = df.sort_values('Total Fat', ascending=False).head(10)

fat_data = fat_data.set_index('Item')

fat_columns = [
    'Calories from Fat',
    'Total Fat',
    'Saturated Fat',
    'Trans Fat'
]

fat_data[fat_columns].plot(
    kind='bar',
    figsize=(14,6)
)

plt.title("Top 10 Menu Items - Fat Related Analysis")
plt.xlabel("Menu Items")
plt.ylabel("Values")
plt.xticks(rotation=45, ha='right')
plt.legend(title="Fat Metrics")
plt.tight_layout()
plt.show()

cholesterol = df.sort_values('Total Fat', ascending=False).head(10)

plt.figure(figsize=(10,5))
plt.bar(top_calories['Item'], top_calories['Cholesterol'])

plt.title("Top 10 Highest Cholesterol Items")
plt.xlabel("Menu Items")
plt.ylabel("Cholesterol")

plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

sodium = df.sort_values('Total Fat', ascending=False).head(10)

plt.figure(figsize=(10,5))
plt.bar(top_calories['Item'], top_calories['Sodium'])

plt.title("Top 10 Highest Sodium Items")
plt.xlabel("Menu Items")
plt.ylabel("Sodium")

plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

carbohydrates = df.sort_values('Total Fat', ascending=False).head(10)

plt.figure(figsize=(10,5))
plt.bar(top_calories['Item'], top_calories['Carbohydrates'])

plt.title("Top 10 Highest Carbohydrates Items")
plt.xlabel("Menu Items")
plt.ylabel("Carbohydrates")

plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

dietary_fiber = df.sort_values('Total Fat', ascending=False).head(10)

plt.figure(figsize=(10,5))
plt.bar(top_calories['Item'], top_calories['Dietary Fiber'])

plt.title("Top 10 Highest Dietary Fiber Items")
plt.xlabel("Menu Items")
plt.ylabel("Dietary Fiber")

plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


sugars = df.sort_values('Total Fat', ascending=False).head(10)

plt.figure(figsize=(10,5))
plt.bar(top_calories['Item'], top_calories['Sugars'])

plt.title("Top 10 Highest Sugars Items")
plt.xlabel("Menu Items")
plt.ylabel("Sugars")

plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


protein = df.sort_values('Total Fat', ascending=False).head(10)

plt.figure(figsize=(10,5))
plt.bar(top_calories['Item'], top_calories['Protein'])

plt.title("Top 10 Highest Protein Items")
plt.xlabel("Menu Items")
plt.ylabel("Protein")

plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Correlaton Analysis
selected_columns = [
    'Calories',
    'Calories from Fat',
    'Total Fat',
    'Saturated Fat',
    'Cholesterol',
    'Sodium',
    'Carbohydrates',
    'Sugars',
    'Protein'
]

corr = df[selected_columns].corr()

plt.figure(figsize=(10,8))

sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    square=True
)

plt.title("Nutritional Features Correlation Heatmap")
plt.tight_layout()
plt.show()