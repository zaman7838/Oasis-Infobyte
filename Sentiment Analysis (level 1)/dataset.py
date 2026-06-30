import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from wordcloud import WordCloud
from sklearn.svm import LinearSVC

nltk.download('stopwords')

df = pd.read_csv(r"C:\Zaman_btech\Zaman_btech\Sentiment Analysis (level 1)\Twitter_Data.csv")
print("\nFirst 5 rows: ")
print(df.head())
print("\nDataset Information: ")
print(df.info())
print("\nShape: ")
print(df.shape)
print("\nCloumn Names: ")
print(df.columns)
print("\nMissing Values: ")
print(df.isnull().sum())
df.dropna(inplace=True)
print("\nDuplicate Records: ")
print(df.duplicated())
print("\nData Types: ")
print(df.dtypes)


# Sentiment Distribution
print(df['category'].value_counts())
df["category"].value_counts().plot(kind="bar", figsize=(6,4))
plt.title("Sentiment Distribution")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()

df["category"].value_counts().plot(kind="pie", autopct = "%1.1f%%", figsize=(6,6))
plt.title("Sentiment Percentage")
plt.ylabel("")
plt.show()
# Tweet Length Analysis
df["text_length"] = df["clean_text"].apply(len)
print(df["text_length"].describe())
plt.figure(figsize=(8,5))
plt.hist(df["text_length"], bins=30)
plt.title("Tweet Length Distribution")
plt.xlabel("Tweet Length")
plt.ylabel("Frequency")
plt.show()
# Positive Tweets
positive_text = " ".join(df[df["category"]==1]["clean_text"])
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(positive_text)
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Positive Tweets Word Cloud")
plt.show()
# Negative Tweets
negative_text = " ".join(df[df["category"]==0]["clean_text"])
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(negative_text)
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Negative Tweets Word Cloud")
plt.show()

# Feature Engineering
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
x = tfidf.fit_transform(df["clean_text"])
y = df["category"]
# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Naive Bayes Model
nb_model = MultinomialNB()
nb_model.fit(x_train, y_train)
nb_pred = nb_model.predict(x_test)
nb_acc = accuracy_score(y_test, nb_pred)
print("Naive Bayes Accuracy:", nb_acc)
# SVM Model
svm_model = LinearSVC()
svm_model.fit(x_train, y_train)
svm_pred = svm_model.predict(x_test)
svm_acc = accuracy_score(y_test, svm_pred)
print("SVM Accuracy:",svm_acc)
# Model Comparison
models = ["Naive Bayes","SVM"]
accuracy = [nb_acc, svm_acc]
plt.figure(figsize=(6,4))
plt.bar(models, accuracy)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()
# Classification Report
print(classification_report(y_test, svm_pred))
# Confusion Matrix
cm = confusion_matrix(y_test, svm_pred)
plt.figure(figsize=(6,4))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.show()
# Custom Prediction
sample = ["This product is amazing and fantastic"]
sample_vector = tfidf.transform(sample)
prediction = svm_model.predict(sample_vector)
print(prediction)