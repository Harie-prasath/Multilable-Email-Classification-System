import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List

# Use this variable for the text column name
TEXT_COL = 'Interaction content'

def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV data from the given filepath."""
    return pd.read_csv(filepath)

def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, remove non-alphanumerics."""
    import re
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text cleaning to the text column."""
    df = df.copy()
    df[TEXT_COL] = df[TEXT_COL].astype(str).apply(clean_text)
    return df

def encode_labels(df: pd.DataFrame, label_cols: List[str]) -> Tuple[pd.DataFrame, dict]:
    """Encode label columns and return encoders for inverse transform."""
    encoders = {}
    for col in label_cols:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders

def extract_features(df: pd.DataFrame, vectorizer: TfidfVectorizer = None) -> Tuple[np.ndarray, TfidfVectorizer]:
    """Extract TF-IDF features from the text column."""
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(df[TEXT_COL])
    else:
        X = vectorizer.transform(df[TEXT_COL])
    return X, vectorizer

def split_data(X, y_intent, y_tone, y_resolution, test_size=0.2, random_state=42):
    """Split data for training and testing."""
    return train_test_split(
        X, y_intent, y_tone, y_resolution, test_size=test_size, random_state=random_state
    ) 