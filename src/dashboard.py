import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import load_feedback_csv
from src.preprocess import simple_clean, make_tfidf
from src.analysis import analyze_dataframe
import numpy as np

st.set_page_config(layout='wide', page_title='Student Feedback Dashboard')

st.title('Student Feedback Text Analysis Dashboard')

with st.sidebar:
    st.header('Load data')
    uploaded = st.file_uploader('Upload CSV with `feedback` column', type=['csv'])
    use_sample = st.button('Use sample data')

if use_sample:
    df = load_feedback_csv('data/sample_feedback.csv')
elif uploaded is not None:
    df = load_feedback_csv(uploaded)
else:
    st.info('Upload a CSV or click "Use sample data".')
    st.stop()

st.subheader('Raw / Anonymized feedback (preview)')
st.dataframe(df[['id', 'feedback_anonymized']].head(50))

# Preprocess + vectorize
texts = simple_clean(df['feedback_anonymized'].astype(str).tolist())
tfidf_matrix, vect = make_tfidf(texts, max_features=500)

# Analysis
n_clusters = st.sidebar.slider('Number of themes (clusters)', 1, 8, 4)
analyzed_df, top_terms, method = analyze_dataframe(df, vectorizer=vect, tfidf_matrix=tfidf_matrix, n_clusters=n_clusters)

st.sidebar.markdown(f'**Sentiment method:** {method}')

st.subheader('Sentiment distribution')
fig, ax = plt.subplots()
ax.hist(analyzed_df['sentiment'], bins=15)
ax.set_xlabel('Sentiment score')
ax.set_ylabel('Count')
st.pyplot(fig)

st.subheader('Top themes (by cluster)')
for k, terms in top_terms.items():
    st.markdown(f'**Theme {k}**: {", ".join(terms)}')

st.subheader('Feedback by theme')
sel = st.selectbox('Choose theme', options=sorted(analyzed_df['theme'].unique()))
subset = analyzed_df[analyzed_df['theme'] == sel].sort_values('sentiment')
st.write(f'Count: {len(subset)}')
st.table(subset[['id', 'feedback_anonymized', 'sentiment']].head(50))

st.markdown('---')
st.write('This dashboard is intentionally simple to show reproducible analysis in a student project. Replace or extend components (sentiment model, topic extraction, visualization) for stronger results.')
