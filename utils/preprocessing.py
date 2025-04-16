import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim.utils import simple_preprocess

# Ensure necessary NLTK data is downloaded
nltk_resources = ['stopwords', 'wordnet', 'punkt']
for resource in nltk_resources:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource)

def preprocess_text(text, stem=True):
    """
    Preprocess the input text by removing non-alphabetic characters,
    converting to lowercase, and performing stemming/lemmatization
    
    Args:
        text (str): Input text to preprocess
        stem (bool): If True, use stemming; otherwise, use lemmatization
    
    Returns:
        str: Preprocessed text
    """
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    
    if stem:
        ps = PorterStemmer()
        text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    else:
        lemmatizer = WordNetLemmatizer()
        text = [lemmatizer.lemmatize(word) for word in text if word not in stopwords.words('english')]
    
    return ' '.join(text)

def get_avg_word2vec_vectors(text, model, dim=100):
    """
    Convert text to average Word2Vec vector
    
    Args:
        text (str): Input text
        model: Word2Vec model
        dim (int): Dimensions of the Word2Vec vectors
    
    Returns:
        numpy.ndarray: Average Word2Vec vector
    """
    tokens = simple_preprocess(text)
    valid_vecs = [model.wv[word] for word in tokens if word in model.wv]
    
    if valid_vecs:
        return np.mean(valid_vecs, axis=0).reshape(1, -1)
    else:
        return np.zeros((1, dim))
    

def preprocess_kindle_review(text):
    """
    Preprocess Kindle review text
    
    Args:
        text (str): Raw review text
        
    Returns:
        str: Preprocessed text
    """
    import re
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
    # Download NLTK resources if needed
    try:
        stopwords.words('english')
    except LookupError:
        import nltk
        nltk.download('stopwords')
        nltk.download('wordnet')
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Preprocessing steps
    text = str(text).lower()
    text = re.sub('[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'(http|https|ftp|ssh)://\S+', '', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    
    return text.strip()