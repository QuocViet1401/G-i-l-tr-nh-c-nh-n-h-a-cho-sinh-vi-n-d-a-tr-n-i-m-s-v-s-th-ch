import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

def process_data(df: pd.DataFrame, score_columns: list) -> tuple:
    df[score_columns] = df[score_columns].fillna(df[score_columns].mean())
    df['Combined Interests'] = df['Trang yêu thích'] + '; ' + df['Nhóm tham gia']

    tfidf_vectorizer = TfidfVectorizer(max_features=100)
    text_features = tfidf_vectorizer.fit_transform(df['Combined Interests']).toarray()

    scaler = StandardScaler()
    normalized_scores = scaler.fit_transform(df[score_columns])

    features = np.hstack((normalized_scores, text_features))
    return df, features, tfidf_vectorizer, scaler

def process_personal_data(personal_df: pd.DataFrame, score_columns: list, manual_pages: str, manual_groups: str,
                          tfidf_vectorizer=None, scaler=None) -> tuple:
    personal_df['Trang yêu thích'] = [manual_pages]
    personal_df['Nhóm tham gia'] = [manual_groups]
    personal_df['Combined Interests'] = personal_df['Trang yêu thích'] + '; ' + personal_df['Nhóm tham gia']

    if tfidf_vectorizer is None:
        tfidf_vectorizer = TfidfVectorizer(max_features=100)
    personal_text_features = tfidf_vectorizer.transform(personal_df['Combined Interests']).toarray()

    if scaler is None:
        scaler = StandardScaler()
    personal_normalized = scaler.transform(personal_df[score_columns]) if hasattr(scaler, 'transform') else scaler.fit_transform(personal_df[score_columns])

    personal_features = np.hstack((personal_normalized, personal_text_features))
    return personal_df, personal_features