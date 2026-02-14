# Student Feedback Text Analysis Dashboard

**Academic project** — A complete, runnable Streamlit dashboard + analysis pipeline to analyze anonymous student feedback.

## Features
- Anonymizes student feedback (removes emails & IDs)
- Basic NLP: cleaning, TF-IDF, clustering (themes)
- Sentiment analysis (uses NLTK VADER; fallback included)
- Streamlit dashboard for visualization and exploration
- Sample dataset + generator script
- Clear project structure and `.gitignore`

## Quick start (VS Code)
1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .venv\Scripts\activate    # Windows (PowerShell)
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (NLTK setup) Download VADER lexicon:
   ```bash
   python -m nltk.downloader vader_lexicon
   ```
4. Run the Streamlit dashboard:
   ```bash
   streamlit run src/dashboard.py --server.port 8501
   ```

## Files
- `src/` — Python source: preprocessing, analysis, dashboard
- `data/sample_feedback.csv` — Sample CSV (id, feedback)
- `requirements.txt` — Python deps
- `.gitignore` — sensible ignores for GitHub
- `LICENSE` — MIT

## Notes
- The pipeline aims to be simple and clear for an academic demo.
- To improve accuracy, replace simple clustering with advanced topic models (e.g., LDA) and use spaCy for better preprocessing.
