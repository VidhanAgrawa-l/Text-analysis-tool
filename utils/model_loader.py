# import os
# import pickle
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO,
#                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# class ModelLoader:
#     """
#     Utility class for loading and managing text classification models
#     """
    
#     def __init__(self, models_dir="models"):
#         """
#         Initialize ModelLoader
        
#         Args:
#             models_dir (str): Directory containing model files
#         """
#         self.models_dir = models_dir
#         self.models = {
#             "bow": {"model": None, "vectorizer": None},
#             "tfidf": {"model": None, "vectorizer": None},
#             "word2vec": {"model": None, "w2v": None}
#         }
#         self.load_all_models()


  

    
#     def load_all_models(self):
#         """Load all available models and vectorizers"""
#         try:
#             # Load BOW model and vectorizer
#             self.models["bow"]["model"] = self._load_pickle("bow_model.pkl")
#             self.models["bow"]["vectorizer"] = self._load_pickle("bow_vectorizer.pkl")
            
#             # Load TF-IDF model and vectorizer
#             self.models["tfidf"]["model"] = self._load_pickle("tfidf_model.pkl")
#             self.models["tfidf"]["vectorizer"] = self._load_pickle("tfidf_vectorizer.pkl")
            
#             # Load Word2Vec model and Word2Vec embeddings
#             self.models["word2vec"]["model"] = self._load_pickle("w2v_model.pkl")
#             self.models["word2vec"]["w2v"] = self._load_pickle("w2v_word2vec.pkl")
            
#             logger.info("All models loaded successfully")
#         except Exception as e:
#             logger.error(f"Error loading models: {e}")
    
#     def _load_pickle(self, filename):
#         """Load a pickle file from the models directory"""
#         filepath = os.path.join(self.models_dir, filename)
#         if not os.path.exists(filepath):
#             logger.warning(f"Model file not found: {filepath}")
#             return None
            
#         with open(filepath, 'rb') as f:
#             return pickle.load(f)
    
#     def predict(self, text, model_type="tfidf"):
#         """
#         Make prediction using specified model type
        
#         Args:
#             text (str): Preprocessed text for prediction
#             model_type (str): Model type to use ('bow', 'tfidf', or 'word2vec')
            
#         Returns:
#             tuple: (prediction label, probability)
#         """
#         if model_type not in self.models:
#             raise ValueError(f"Unsupported model type: {model_type}")
            
#         model_dict = self.models[model_type]
        
#         if None in model_dict.values():
#             raise ValueError(f"Model {model_type} is not loaded properly")
            
#         if model_type == "bow":
#             vectorized = model_dict["vectorizer"].transform([text]).toarray()
#             prediction = model_dict["model"].predict(vectorized)[0]
#             probability = max(model_dict["model"].predict_proba(vectorized)[0])
        
#         elif model_type == "tfidf":
#             vectorized = model_dict["vectorizer"].transform([text]).toarray()
#             prediction = model_dict["model"].predict(vectorized)[0]
#             probability = max(model_dict["model"].predict_proba(vectorized)[0])
        
#         elif model_type == "word2vec":
#             from utils.preprocessing import get_avg_word2vec_vectors
#             vectorized = get_avg_word2vec_vectors(text, model_dict["w2v"])
#             prediction = model_dict["model"].predict(vectorized)[0]
#             probability = max(model_dict["model"].predict_proba(vectorized)[0])
        
#         return int(prediction), float(probability)


import os
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Utility class for loading and managing text classification models
    """
    
    def __init__(self, models_dir="models"):
        """
        Initialize ModelLoader
        
        Args:
            models_dir (str): Directory containing model files
        """
        self.models_dir = models_dir
        self.models = {
            "bow": {"model": None, "vectorizer": None},
            "tfidf": {"model": None, "vectorizer": None},
            "word2vec": {"model": None, "w2v": None}
        }
        # Kindle models dictionary
        self.kindle_models = {
            "bow": {"model": None, "vectorizer": None},
            "tfidf": {"model": None, "vectorizer": None}
        }
        self.load_all_models()

    def load_all_models(self):
        """Load all available models and vectorizers"""
        try:
            # Load SMS spam models
            self.models["bow"]["model"] = self._load_pickle("bow_model.pkl")
            self.models["bow"]["vectorizer"] = self._load_pickle("bow_vectorizer.pkl")
            
            self.models["tfidf"]["model"] = self._load_pickle("tfidf_model.pkl")
            self.models["tfidf"]["vectorizer"] = self._load_pickle("tfidf_vectorizer.pkl")
            
            self.models["word2vec"]["model"] = self._load_pickle("w2v_model.pkl")
            self.models["word2vec"]["w2v"] = self._load_pickle("w2v_word2vec.pkl")
            
            # Load Kindle models
            self.kindle_models["bow"]["model"] = self._load_pickle("kindle/bow_model.pkl")
            self.kindle_models["bow"]["vectorizer"] = self._load_pickle("kindle/bow_vectorizer.pkl")
            
            self.kindle_models["tfidf"]["model"] = self._load_pickle("kindle/tfidf_model.pkl")
            self.kindle_models["tfidf"]["vectorizer"] = self._load_pickle("kindle/tfidf_vectorizer.pkl")
            
            logger.info("All models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def _load_pickle(self, filename):
        """Load a pickle file from the models directory"""
        filepath = os.path.join(self.models_dir, filename)
        if not os.path.exists(filepath):
            logger.warning(f"Model file not found: {filepath}")
            return None
            
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def predict(self, text, model_type="tfidf"):
        """
        Make prediction using specified model type
        
        Args:
            text (str): Preprocessed text for prediction
            model_type (str): Model type to use ('bow', 'tfidf', or 'word2vec')
            
        Returns:
            tuple: (prediction label, probability)
        """
        if model_type not in self.models:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        model_dict = self.models[model_type]
        
        if None in model_dict.values():
            raise ValueError(f"Model {model_type} is not loaded properly")
            
        if model_type == "bow":
            vectorized = model_dict["vectorizer"].transform([text]).toarray()
            prediction = model_dict["model"].predict(vectorized)[0]
            probability = max(model_dict["model"].predict_proba(vectorized)[0])
        
        elif model_type == "tfidf":
            vectorized = model_dict["vectorizer"].transform([text]).toarray()
            prediction = model_dict["model"].predict(vectorized)[0]
            probability = max(model_dict["model"].predict_proba(vectorized)[0])
        
        elif model_type == "word2vec":
            from utils.preprocessing import get_avg_word2vec_vectors
            vectorized = get_avg_word2vec_vectors(text, model_dict["w2v"])
            prediction = model_dict["model"].predict(vectorized)[0]
            probability = max(model_dict["model"].predict_proba(vectorized)[0])
        
        return int(prediction), float(probability)
    
    def predict_kindle_sentiment(self, text, model_type="tfidf"):
        """
        Make sentiment prediction for Kindle reviews using specified model type
        
        Args:
            text (str): Preprocessed text for prediction
            model_type (str): Model type to use ('bow' or 'tfidf')
            
        Returns:
            tuple: (prediction label, probability array)
        """
        if model_type not in self.kindle_models:
            raise ValueError(f"Unsupported Kindle model type: {model_type}")
            
        model_dict = self.kindle_models[model_type]
        
        if None in model_dict.values():
            raise ValueError(f"Kindle model {model_type} is not loaded properly")
            
        vectorized = model_dict["vectorizer"].transform([text]).toarray()
        prediction = model_dict["model"].predict(vectorized)[0]
        probabilities = model_dict["model"].predict_proba(vectorized)[0]
        
        return int(prediction), probabilities

def load_kindle_model(model_type):
    """
    Function for directly loading Kindle models without instantiating the class
    
    Args:
        model_type (str): Model type to load ('bow' or 'tfidf')
        
    Returns:
        tuple: (model, vectorizer)
    """
    if model_type == "bow":
        model_path = os.path.join("models", "kindle", "bow_model.pkl")
        vectorizer_path = os.path.join("models", "kindle", "bow_vectorizer.pkl")
    elif model_type == "tfidf":
        model_path = os.path.join("models", "kindle", "tfidf_model.pkl")
        vectorizer_path = os.path.join("models", "kindle", "tfidf_vectorizer.pkl")
    else:
        raise ValueError(f"Model type {model_type} not supported for Kindle analysis")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    return model, vectorizer