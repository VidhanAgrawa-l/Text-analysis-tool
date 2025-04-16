# train_and_save_models.py

import pandas as pd
import re
import nltk
import pickle
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

import gensim.downloader as api

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

# Load Dataset
messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', names=["label", "message"])

# Encode labels
y = pd.get_dummies(messages['label'], drop_first=True).values.ravel()

# Preprocessing
def preprocess_text(text, stem=True):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    if stem:
        ps = PorterStemmer()
        text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    else:
        lemmatizer = WordNetLemmatizer()
        text = [lemmatizer.lemmatize(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

corpus_stem = [preprocess_text(msg, stem=True) for msg in messages['message']]
corpus_lem = [preprocess_text(msg, stem=False) for msg in messages['message']]

# ------------------------ BOW ------------------------
cv = CountVectorizer(max_features=2500, ngram_range=(1, 2))
X_bow = cv.fit_transform(corpus_stem).toarray()
X_train, X_test, y_train, y_test = train_test_split(X_bow, y, test_size=0.2)
model_bow = MultinomialNB().fit(X_train, y_train)
print("\n[BOW] Accuracy:", accuracy_score(y_test, model_bow.predict(X_test)))
print(classification_report(y_test, model_bow.predict(X_test)))

with open('models/bow_vectorizer.pkl', 'wb') as f: pickle.dump(cv, f)
with open('models/bow_model.pkl', 'wb') as f: pickle.dump(model_bow, f)

# ------------------------ TF-IDF ------------------------
tv = TfidfVectorizer(max_features=2500, ngram_range=(1, 2))
X_tfidf = tv.fit_transform(corpus_stem).toarray()
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2)
model_tfidf = MultinomialNB().fit(X_train, y_train)
print("\n[TF-IDF] Accuracy:", accuracy_score(y_test, model_tfidf.predict(X_test)))
print(classification_report(y_test, model_tfidf.predict(X_test)))

with open('models/tfidf_vectorizer.pkl', 'wb') as f: pickle.dump(tv, f)
with open('models/tfidf_model.pkl', 'wb') as f: pickle.dump(model_tfidf, f)

# ------------------------ Word2Vec ------------------------
words = []
for sent in corpus_lem:
    sent_tokens = nltk.sent_tokenize(sent)
    for s in sent_tokens:
        words.append(simple_preprocess(s))

word2vec_model = Word2Vec(sentences=words, vector_size=100, window=5, min_count=1)

# Average Word Vectors
def get_avg_word2vec_vectors(corpus, model, dim=100):
    vectors = []
    for text in corpus:
        tokens = simple_preprocess(text)
        valid_vecs = [model.wv[word] for word in tokens if word in model.wv]
        if valid_vecs:
            vectors.append(np.mean(valid_vecs, axis=0))
        else:
            vectors.append(np.zeros(dim))
    return np.array(vectors)

X_w2v = get_avg_word2vec_vectors(corpus_lem, word2vec_model)
X_train, X_test, y_train, y_test = train_test_split(X_w2v, y, test_size=0.2)
from sklearn.linear_model import LogisticRegression
model_w2v = LogisticRegression(max_iter=1000).fit(X_train, y_train)

print("\n[Word2Vec] Accuracy:", accuracy_score(y_test, model_w2v.predict(X_test)))
print(classification_report(y_test, model_w2v.predict(X_test)))

with open('models/w2v_model.pkl', 'wb') as f: pickle.dump(model_w2v, f)
with open('models/w2v_word2vec.pkl', 'wb') as f: pickle.dump(word2vec_model, f)

print("\nAll models and vectorizers saved to 'models/' folder.")
