import re
from typing import List
import pandas as pd

EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+')
ID_RE = re.compile(r'\b\d{5,}\b')  # crude student ID removal

def anonymize_text(text: str) -> str:
    """Remove emails, long numeric ids and extra whitespace."""
    if not isinstance(text, str):
        return ''
    t = EMAIL_RE.sub('[EMAIL]', text)
    t = ID_RE.sub('[ID]', t)
    t = ' '.join(t.split())
    return t

def load_feedback_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'id' not in df.columns:
        df.insert(0, 'id', range(1, len(df) + 1))
    if 'feedback' not in df.columns:
        raise ValueError('CSV must contain a `feedback` column.')
    df['feedback_anonymized'] = df['feedback'].astype(str).apply(anonymize_text)
    return df
