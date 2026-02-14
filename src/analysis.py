import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.exceptions import NotFittedError

# fallback if VADER not available
SMALL_LEXICON = {
    'good': 2, 'great': 3, 'excellent': 3, 'helpful': 2, 'easy': 1,
    'bad': -2, 'difficult': -2, 'unclear': -2, 'harsh': -2, 'slow': -1
}

def sentiment_scores(texts):
    try:
        sid = SentimentIntensityAnalyzer()
        scores = [sid.polarity_scores(t)['compound'] for t in texts]
        return scores, 'vader'
    except Exception:
        # simple fallback
        def score_text(t):
            s = 0
            words = t.split()
            for w in words:
                s += SMALL_LEXICON.get(w, 0)
            return max(-1, min(1, s/5))
        return [score_text(t) for t in texts], 'small_lexicon'

def cluster_themes(tfidf_matrix, vectorizer, n_clusters=4):
    # KMeans over TF-IDF to get clusters (themes)
    if tfidf_matrix.shape[0] < n_clusters:
        n_clusters = max(1, tfidf_matrix.shape[0])
    km = KMeans(n_clusters=n_clusters, random_state=42)
    km.fit(tfidf_matrix)
    labels = km.labels_
    centers = km.cluster_centers_
    # extract top terms per cluster
    terms = vectorizer.get_feature_names_out()
    top_terms = {}
    for i in range(km.n_clusters):
        center = centers[i]
        top_idx = np.argsort(center)[-8:][::-1]
        top_terms[i] = [terms[j] for j in top_idx]
    return labels, top_terms, km

def analyze_dataframe(df, vectorizer=None, tfidf_matrix=None, n_clusters=4):
    texts = df['feedback_anonymized'].astype(str).tolist()
    if vectorizer is None or tfidf_matrix is None:
        raise NotFittedError('vectorizer and tfidf_matrix must be supplied')
    scores, method = sentiment_scores(texts)
    df = df.copy()
    df['sentiment'] = scores
    labels, top_terms, km = cluster_themes(tfidf_matrix, vectorizer, n_clusters=n_clusters)
    df['theme'] = labels
    return df, top_terms, method
