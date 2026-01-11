"""
model.py - Lightweight ML model with interpretable feature weights
Detects unclear requirements using keyword-based logistic regression.
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

# Synthetic mini dataset
TRAIN_SENTENCES = [
    "The system shall be fast and scalable.",
    "The UI should be user-friendly and flexible.",
    "The system shall respond in under 2 seconds.",
    "The process should handle 1000 records within 5 seconds.",
    "The application must be reliable and robust.",
    "The system must store 10GB of logs daily.",
]
y = [1, 1, 0, 0, 1, 0]  # 1 = Unclear, 0 = Clear

# Create model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(TRAIN_SENTENCES)
clf = LogisticRegression()
clf.fit(X, y)

# Extract feature importance
feature_names = np.array(vectorizer.get_feature_names_out())
coeffs = clf.coef_[0]

# Predict unclear probability
def predict_unclear(sentence: str) -> float:
    vec = vectorizer.transform([sentence])
    return float(clf.predict_proba(vec)[0][1])

# Explain contribution of words
def explain_features(sentence: str, top_n=5):
    vec = vectorizer.transform([sentence])
    feature_idx = vec.nonzero()[1]
    words = feature_names[feature_idx]
    weights = coeffs[feature_idx]
    importance = sorted(zip(words, weights), key=lambda x: abs(x[1]), reverse=True)
    return importance[:top_n]