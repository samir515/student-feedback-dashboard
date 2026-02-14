import re
from sklearn.feature_extraction.text import TfidfVectorizer

def simple_clean(texts):
    cleaned = []
    for t in texts:
        t = t.lower()
        t = re.sub(r'[^a-z0-9\s]', ' ', t)
        t = re.sub(r'\s+', ' ', t).strip()
        cleaned.append(t)
    return cleaned

def make_tfidf(texts, max_features=1000):
    vect = TfidfVectorizer(max_features=max_features, stop_words='english')
    X = vect.fit_transform(texts)
    return X, vect
