import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data Loading
df = pd.read_csv(r"C:\Zaman_btech\Zaman_btech\EDA on Retail Sales Data (level 1)\retail_sales_dataset.csv")

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

df['Date'] = pd.to_datetime(df['Date']) # Date column in Datetime format

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

# Time Series Analysis

df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

monthly_sales = df.groupby('Month')['Total Amount'].sum()
print("\nMonthly Sales: ")
print(monthly_sales)

plt.figure(figsize=(10,5))
plt.plot(monthly_sales.index,monthly_sales.values,marker='o')
plt.title("Monthly Sales Trends")
plt.xlabel("Month")
plt.ylabel("Total Amount")
plt.grid(True)
plt.show()

print("\nHighest Sales Month:", monthly_sales.idxmax())
print("\nLowest Sales Month:", monthly_sales.idxmin())

# Product Analysis
product_sales = df.groupby('Product Category')['Total Amount'].sum()
print("\nProduct Sales: ")
print(product_sales.sort_values(ascending=False))
top_products = product_sales.sort_values(ascending=False).head(10)
print("\nTop Products: ")
print(top_products)

plt.figure(figsize=(10,6))
top_products.plot(kind='bar')
plt.title("Top Selling Products")
plt.xlabel("Products")
plt.ylabel("Total Amount")
plt.show()

print("\nBest Selling Product: ", product_sales.idxmax())
print("\nLowest Selling Product: ",product_sales.idxmin())

# Customer Analysis
if 'Customer ID' in df.columns:
    customer_sales = df.groupby('Customer ID')['Total Amount'].sum()
    print("\nCustomer Sales: ")
    print(customer_sales.sort_values(ascending=False))
    top_customers = customer_sales.sort_values(ascending=False).head(10)
    print("\nTop 10 Customers: ")
    print(top_customers)

    plt.figure(figsize=(10,6))
    top_customers.plot(kind='bar')
    plt.title("Top 10 Customers")
    plt.xlabel("Customers")
    plt.ylabel("Total Amount")
    plt.show()

    # Correlaton Analysis
    print("\nCorrelation Matrix: ")
    print(df.corr(numeric_only=True))
    plt.figure(figsize=(8,6))
    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap='coolwarm'
    )
    plt.title("Correlation Heatmap")
    plt.show()

# Recommendations
print("\n===== BUSINESS RECOMMENDATIONS =====")

print("1. Focus marketing campaigns on the highest-selling product categories.")
print("2. Increase inventory for products with consistently high sales.")
print("3. Offer discounts and promotions during low-sales months.")
print("4. Target customer groups that contribute the most revenue.")
print("5. Introduce loyalty programs to improve customer retention.")
print("6. Use seasonal trends to plan stock and marketing strategies.")
print("7. Analyze underperforming products and consider promotional offers.")
print("8. Monitor monthly sales regularly to identify growth opportunities.")


