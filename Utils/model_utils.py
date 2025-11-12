import os, pickle
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = "models/neomind_model.pkl"

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    # First-time model
    return {
        "vectorizer": TfidfVectorizer(),
        "classifier": SGDClassifier(max_iter=1000, tol=1e-3),
        "label_encoder": LabelEncoder(),
        "trained": False
    }

def save_model(model):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
