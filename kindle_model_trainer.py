# kindle_model_trainer.py

import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# --- Load Dataset ---
df = pd.read_csv('all_kindle_review.csv')[['reviewText', 'rating']]
df['rating'] = df['rating'].apply(lambda x: 0 if x < 3 else 1)

# --- Preprocessing ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'(http|https|ftp|ssh)://\S+', '', text)
    # text = BeautifulSoup(text, 'lxml').get_text()
    text = " ".join([word for word in text.split() if word not in stop_words])
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    return text.strip()

df['reviewText'] = df['reviewText'].astype(str).apply(preprocess)

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(df['reviewText'], df['rating'], test_size=0.2, random_state=42)

# --- BoW ---
bow_vectorizer = CountVectorizer()
X_train_bow = bow_vectorizer.fit_transform(X_train).toarray()
X_test_bow = bow_vectorizer.transform(X_test).toarray()
bow_model = GaussianNB().fit(X_train_bow, y_train)

# --- TF-IDF ---
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = tfidf_vectorizer.transform(X_test).toarray()
tfidf_model = GaussianNB().fit(X_train_tfidf, y_train)

# --- Save Models and Vectorizers ---
output_dir = 'models/kindle'  # You can change this
import os
os.makedirs(output_dir, exist_ok=True)

with open(f'{output_dir}/bow_model.pkl', 'wb') as f:
    pickle.dump(bow_model, f)
with open(f'{output_dir}/bow_vectorizer.pkl', 'wb') as f:
    pickle.dump(bow_vectorizer, f)

with open(f'{output_dir}/tfidf_model.pkl', 'wb') as f:
    pickle.dump(tfidf_model, f)
with open(f'{output_dir}/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

print("✅ Models and vectorizers saved successfully!")
