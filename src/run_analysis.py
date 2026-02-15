# src/run_analysis.py
import os
from src.utils import load_feedback_csv
from src.preprocess import simple_clean, make_tfidf
from src.analysis import sentiment_scores, cluster_themes
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = os.path.join("data", "sample_feedback.csv") 

def main(path=DATA_PATH, n_clusters=4, outdir="outputs"):
    os.makedirs(outdir, exist_ok=True)

    df = load_feedback_csv(path)
    texts = simple_clean(df['feedback_anonymized'].astype(str).tolist())
    tfidf_matrix, vect = make_tfidf(texts, max_features=500)

    # Sentiment
    scores, method = sentiment_scores(df['feedback_anonymized'].astype(str).tolist())
    df['sentiment'] = scores

    # Clustering
    labels, top_terms, km = cluster_themes(tfidf_matrix, vect, n_clusters=n_clusters)
    df['theme'] = labels

    # Save the annotated CSV
    out_csv = os.path.join(outdir, "feedback_annotated.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved annotated CSV to: {out_csv}")

    # Save histogram
    fig, ax = plt.subplots()
    ax.hist(df['sentiment'], bins=15)
    ax.set_xlabel("Sentiment score")
    ax.set_ylabel("Count")
    hist_path = os.path.join(outdir, "sentiment_histogram.png")
    fig.savefig(hist_path, bbox_inches="tight")
    print(f"Saved sentiment histogram to: {hist_path}")

    # Print top terms per theme to console and save
    txt_path = os.path.join(outdir, "top_terms_by_theme.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for k, terms in top_terms.items():
            line = f"Theme {k}: {', '.join(terms)}"
            print(line)
            f.write(line + "\\n")
    print(f"Saved top terms to: {txt_path}")

if __name__ == "__main__":
    main()
